"""Prepare native EVA/human Zarr recordings for the raw-pixel world model.

Inputs explicitly select physical episodes and episode-level splits. S3 reads
reuse the native downloader; outputs are new immutable files. No annotation
inference, training, resampling, image flips, or source writes occur here.
"""
from __future__ import annotations
import argparse
import copy
import os
import shutil
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
import h5py
import numpy as np
from egomimic.rldb.camera_pose import native_camera_world_poses, relative_camera_pose
from egomimic.rldb.world_model_preparation import StreamingMoments, training_identities, ArtifactVerificationMemo
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix, _matrix_to_xyzrot6d


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def immutable_json(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write('\n')


def _array(group, name, frames):
    values = np.asarray(group[name][:frames])
    if len(values) != frames:
        raise ValueError(f'Incomplete native field {name}')
    return values


def native_numeric_arrays(group, cameras, *, maximum_source_gap_s=0.1):
    attrs = dict(group.attrs)
    n = int(attrs['total_frames'])
    embodiment = attrs['embodiment']
    if n < 1:
        raise ValueError('Empty native episode')
    if embodiment == 'eva_bimanual':
        poses, actions = [], []
        for arm in ('left', 'right'):
            # Actual native EVA consumer takes inverse(stored calibration).
            extrinsic = np.asarray(attrs.get('extrinsics', {}).get(arm), np.float64)
            if extrinsic.shape != (4,4):
                raise ValueError('Measured per-arm front camera calibration required')
            for role, target in [('obs', poses), ('cmd', actions)]:
                raw = _array(group, f'{arm}.{role}_ee_pose', n)
                matrix = np.linalg.inv(extrinsic) @ _xyzwxyz_to_matrix(raw)
                gripper = _array(group, f'{arm}.{role}_gripper', n).reshape(n,1)
                target.append(np.concatenate([_matrix_to_xyzrot6d(matrix), gripper], -1))
        pose, action = np.concatenate(poses,-1), np.concatenate(actions,-1)
        fps = float(attrs['fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Positive native frame rate required')
        times = np.arange(n, dtype=np.float64) / fps
        clock = 'native synchronized EVA row index / recorded fps; no invented acquisition timestamps'
        head = None
        pose_valid = np.isfinite(pose).all(axis=1) & (np.abs(pose).max(axis=1) < 1000)
        if not pose_valid.all():
            raise ValueError('Invalid recorded robot Cartesian state')
    elif embodiment == 'human_bimanual':
        pose = np.concatenate([_array(group, f'{arm}.obs_keypoints', n).reshape(n,63)
                               for arm in ('left','right')], -1)
        action = None
        hands = pose.reshape(n,2,63)
        pose_valid = (np.isfinite(hands).all(axis=2) & (np.abs(hands).max(axis=2) < 1000)
                      & (np.abs(hands).sum(axis=2) > 1e-9)).all(axis=1)
        pose = np.where(pose_valid[:,None],pose,0)
        timestamps = _array(group, 'obs_rgb_timestamps_ns', n).reshape(n)
        times = (timestamps - timestamps[0]).astype(np.float64) * 1e-9
        clock = 'recorded obs_rgb_timestamps_ns minus first physical timestamp'
        head = _array(group, 'obs_head_pose', n)
    else:
        raise ValueError(f'Unsupported native embodiment: {embodiment}')
    if not pose_valid.any():
        raise ValueError('No valid measured proprioception in selected episode')
    if not np.isfinite(maximum_source_gap_s) or maximum_source_gap_s <= 0:
        raise ValueError('Positive source-gap threshold required')
    if action is not None and not np.isfinite(action).all():
        raise ValueError('Nonfinite real native command')
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Native clock must strictly increase')
    cuts = np.flatnonzero(np.diff(times) > maximum_source_gap_s) + 1
    boundaries = np.r_[0,cuts,n]
    source_cuts = [[int(a),int(b)] for a,b in zip(boundaries[:-1],boundaries[1:])]
    camera, valid = native_camera_world_poses(embodiment, cameras, n,
        head_pose=head, extrinsics=attrs.get('extrinsics'))
    front_valid = valid[:,cameras.index('front_1')] if 'front_1' in cameras else valid.any(axis=1)
    usable = pose_valid & front_valid
    if embodiment == 'human_bimanual':
        # Native Human keypoint transform: inv(world_from_RGBcamera_t) @
        # world_keypoints_t. Future camera measurements construct targets only;
        # the loader still exposes their unknown diffusion roles, never context.
        from egomimic.rldb.zarr.action_chunk_transforms import ActionChunkCoordinateFrameTransform
        transform = ActionChunkCoordinateFrameTransform('camera','keypoints','camera_keypoints',mode='xyz')
        camera_pose = np.zeros_like(pose)
        for t in np.flatnonzero(usable):
            transformed = transform.transform(dict(camera=head[t],keypoints=pose[t].reshape(42,3)))
            camera_pose[t] = transformed['camera_keypoints'].reshape(126)
        pose = camera_pose
        pose_valid = usable.copy()
    continuity = []
    for a,b in source_cuts:
        inside = usable[a:b]
        starts = np.flatnonzero(inside & ~np.r_[False,inside[:-1]]) + a
        ends = np.flatnonzero(inside & ~np.r_[inside[1:],False]) + a + 1
        continuity.extend([[int(lo),int(hi)] for lo,hi in zip(starts,ends)])
    if not continuity:
        raise ValueError('No jointly measured state/front-camera rows for sampling')
    return dict(embodiment=embodiment, frames=n, time=times, pose=pose.astype(np.float32),
                action=None if action is None else action.astype(np.float32),
                camera_world_pose=camera, camera_pose_valid=valid, clock=clock, pose_valid=pose_valid,
                continuity_segments=continuity,maximum_source_gap_s=maximum_source_gap_s,
                source_cut_segments=source_cuts, usable_frame=usable,
                quality_exclusions=dict(invalid_pose_frames=int((~pose_valid).sum()),
                    invalid_front_camera_frames=int((~front_valid).sum()),
                    excluded_union_frames=int((~usable).sum()), usable_frames=int(usable.sum())))


def prepare(selection_path, destination, *, source_cache, image_size=(160,160), patch_size=16, defer_normalization=False):
    import cv2
    import zarr
    from egomimic.rldb.zarr.zarr_dataset_multi import decode_jpeg_payload
    from egomimic.utils.aws.aws_data_utils import s3_sync_to_local
    selection = selection_path if isinstance(selection_path,dict) else json.loads(Path(selection_path).read_text())
    entries = selection['episodes']
    if (len(image_size) != 2 or any(type(v) is not int or v < 1 for v in image_size)
            or type(patch_size) is not int or patch_size < 1 or any(v % patch_size for v in image_size)):
        raise ValueError('Positive image dimensions divisible by patch size required')
    if len({e['source'] for e in entries}) != len(entries):
        raise ValueError('One physical source cannot appear in multiple split entries')
    if not defer_normalization and {e['embodiment'] for e in entries} != {e['embodiment'] for e in entries if e['split']=='train'}:
        raise ValueError('Every prepared embodiment requires training-only normalization data')
    identities = [(e['embodiment'],e['episode_id']) for e in entries]
    if not entries or len(set(identities)) != len(entries):
        raise ValueError('Explicit nonduplicate physical episode selection required')
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('Prepared manifest already exists; preserve immutable output')
    moments, episodes, pose_schemas, action_schemas, camera_schemas = {}, [], {}, {}, {}
    for entry in entries:
        if entry['split'] not in ('train','val'):
            raise ValueError('Explicit train/val split required')
        eid, embodiment = entry['episode_id'], entry['embodiment']
        if Path(eid).name != eid or eid in ('.','..'):
            raise ValueError('Episode ID must be safe single component')
        source = entry['source']
        if source.startswith('s3://'):
            uri = urlsplit(source)
            local = Path(source_cache)/eid
            s3_sync_to_local(uri.netloc, uri.path.lstrip('/'), local)
        else:
            local = Path(source)
        if not (local/'zarr.json').is_file():
            raise ValueError('Native Zarr v3 metadata missing')
        # One bounded native episode inventory binds all original bytes read.
        native_files = [{'path':str(f.relative_to(local)), 'bytes':f.stat().st_size, 'sha256':sha(f)}
                        for f in sorted(local.rglob('*')) if f.is_file()]
        native_payload = hashlib.sha256(json.dumps(native_files,sort_keys=True).encode()).hexdigest()
        g = zarr.open_group(str(local), mode='r')
        cameras = list(entry['cameras'])
        arrays = native_numeric_arrays(g, cameras)
        if arrays['embodiment'] != embodiment:
            raise ValueError('Source embodiment differs from selection')
        n = arrays['frames']
        numeric = output/f'{eid}-state.h5'
        rgb = output/f'{eid}-rgb.h5'
        with h5py.File(numeric,'x') as f:
            for key in ('time','pose','camera_world_pose','camera_pose_valid'):
                f[key] = arrays[key]
            f['pose_valid'] = arrays['pose_valid']
            if arrays['action'] is not None:
                f['action'] = arrays['action']
                f['action_valid'] = np.ones(n,bool)
        with h5py.File(rgb,'x') as f:
            f['time'] = arrays['time']
            frames = f.create_dataset('rgb',(n,len(cameras),*image_size,3),dtype='u1',
                chunks=(1,1,*image_size,3),compression='lzf')
            for c,name in enumerate(cameras):
                native = g[f'images.{name}']
                for t in range(n):
                    payload = native[t]
                    if isinstance(payload,np.ndarray) and payload.ndim==3 and payload.dtype==np.uint8:
                        image = payload if payload.shape[-1]==3 else np.moveaxis(payload,0,-1)
                    else:
                        image = np.moveaxis(decode_jpeg_payload(payload),0,-1)
                        image = np.rint(image*255).astype(np.uint8)
                    frames[t,c] = cv2.resize(image,(image_size[1],image_size[0]),interpolation=cv2.INTER_AREA)
        native_proof = output/f'{eid}-native.json'
        immutable_json(native_proof,dict(source=source,files=native_files,clock=arrays['clock'],
            image_processing='full_frame_INTER_AREA_no_flip',camera_reference='absolute measured camera matrices; reanchor after window sampling'))
        ep=dict(embodiment=embodiment,episode_id=eid,split=entry['split'],frames=n,
            path=numeric.name,sha256=sha(numeric),rgb_path=rgb.name,rgb_sha256=sha(rgb),
            cameras=cameras,frame='fixed_reference' if arrays['action'] is not None else 'per_frame_rgb_camera',
            pose_kind='measured_pose' if arrays['action'] is not None else 'measured_keypoints',
            continuity_segments=arrays['continuity_segments'],maximum_source_gap_s=arrays['maximum_source_gap_s'],
            source_cut_segments=arrays['source_cut_segments'],quality_exclusions=arrays['quality_exclusions'],
            native_source=dict(source_demo_sha256=native_payload,episode=eid,
                               provenance_path=native_proof.name,provenance_sha256=sha(native_proof)))
        dim=arrays['pose'].shape[-1]
        pose_schemas[embodiment]=dict(dimension=dim,fields=[f'coordinate_{i}' for i in range(dim)],
            representation='bimanual_xyz_rotation6d_gripper_camera' if arrays['action'] is not None else 'bimanual_MANO_per_frame_RGBcamera_keypoints',
            coordinate_reference='fixed_front_camera' if arrays['action'] is not None else 'each_frame_RGBcamera',
            units='meters')
        camera_schemas[embodiment]=dict(dimension=9,fields=['x','y','z','r00','r10','r20','r01','r11','r21'],representation='xyz_rotation6d')
        if arrays['action'] is not None:
            proof=output/f'{eid}-alignment.json'
            immutable_json(proof,dict(schema='pixel-proprio-action-alignment-v1',relation='action_t_applied_to_state_t',
                source_demo_sha256=native_payload,source_episode=eid,producer_sha256=sha(__file__),
                raw_export_metadata_sha256=sha(local/'zarr.json'),raw_payload_sha256=native_payload,
                cache_frame_index=list(range(n)),source_state_index=list(range(n)),source_action_index=list(range(n)),
                basis=dict(cache_index_source='Every native Zarr row preserved without temporal resampling',
                           action_index_source='Native left/right.cmd_ee_pose and cmd_gripper at same row',
                           state_action_relation_source='EVA extractor and eva_to_zarr preserve synchronized cmd/obs arrays with identical frame indices; command target, not finite-difference action')))
            ep['action_provenance']=dict(path=proof.name,sha256=sha(proof))
            action_schemas[embodiment]=dict(dimension=20,fields=[f'command_{i}' for i in range(20)],
                semantics='action_command',value_space='native_commands',representation='native_absolute_cmd_pose_and_gripper_in_fixed_front_camera')
        ep['prepared_file_identity']={key:dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,inode=path.stat().st_ino,uid=path.stat().st_uid) for key,path in [('path',numeric),('rgb_path',rgb)]}
        episodes.append(ep)
        if entry['split']=='train' and not defer_normalization:
            moments.setdefault(f'pose/{embodiment}',StreamingMoments()).update(arrays['pose'][arrays['usable_frame']])
            cam_valid = arrays['camera_pose_valid'] & arrays['usable_frame'][:,None]
            refs = [int(np.flatnonzero(cam_valid[:,c])[0]) if cam_valid[:,c].any() else 0 for c in range(cam_valid.shape[1])]
            relative, valid = relative_camera_pose(arrays['camera_world_pose'],cam_valid,refs)
            moments.setdefault(f'camera/{embodiment}',StreamingMoments()).update(relative[valid])
    if defer_normalization:
        normalization=dict(scope='pending_global_train_only',arrays_normalized=False)
    else:
        norm=output/'normalization.json'
        immutable_json(norm,dict(schema='pixel-proprio-normalization-v1',scope='train_only',
            train_episodes=training_identities(episodes),statistics={k:v.result(.05) for k,v in moments.items()},
            camera_moment_reference='first usable physical episode frame per measured camera, fixed TRAIN-only statistics over usable rows; samples reanchor to first sampled frame'))
        normalization=dict(scope='train_only',arrays_normalized=False,path=norm.name,sha256=sha(norm))
    manifest=dict(schema='pixel-proprio-prepared-v1',episodes=episodes,pose_schemas=pose_schemas,
        action_schemas=action_schemas,camera_pose_schemas=camera_schemas,
        pixel_identity=dict(image_size=list(image_size),patch_size=patch_size,channels=3,patch_order='row_major_hwc',normalization='uint8_to_minus_one_one'),
        normalization=normalization)
    immutable_json(output/'manifest.json',manifest)
    return manifest



def _native_record_source(ep, root):
    proof=ep['native_source']
    path=root/proof['provenance_path']
    if sha(path)!=proof['provenance_sha256']:
        raise ValueError('Prepared source provenance digest mismatch')
    return json.loads(path.read_text())['source']


def merge_prepared_manifests(selection_path, manifest_paths, destination, *, link_caches=False):
    """Publish one train-only normalization across immutable raw episode caches.

    Physical splits come only from the authoritative selection. RGB is referenced,
    never copied or decoded. Deferred manifests cannot be loaded before this gate.
    """
    selection=json.loads(Path(selection_path).read_text())
    entries=selection['episodes']
    selected={(e['embodiment'],e['episode_id']):e for e in entries}
    if len(selected)!=len(entries) or len({e['source'] for e in entries})!=len(entries):
        raise ValueError('Duplicate physical episode/source in split selection')
    if any(e['split'] not in ('train','val') for e in entries):
        raise ValueError('Explicit physical train/val split required')
    output=Path(destination);output.mkdir(parents=True,exist_ok=True)
    episodes,schemas,pixels={},dict(pose_schemas={},action_schemas={},camera_pose_schemas={}),None
    bindings=[]
    for manifest_path in manifest_paths:
        path=Path(manifest_path).resolve();manifest=json.loads(path.read_text());root=path.parent
        if manifest.get('schema')!='pixel-proprio-prepared-v1' or manifest.get('normalization',{}).get('arrays_normalized') is not False:
            raise ValueError('Raw prepared manifest required for global normalization')
        if pixels is None:pixels=manifest['pixel_identity']
        if pixels!=manifest['pixel_identity']:raise ValueError('Mixed pixel processing identities')
        bindings.append(dict(path=str(path),sha256=sha(path)))
        for name,group in schemas.items():
            for emb,item in manifest[name].items():
                if emb in group and group[emb]!=item:raise ValueError('Mixed embodiment schema')
                group[emb]=item
        for original in manifest['episodes']:
            key=original['embodiment'],original['episode_id']
            if key not in selected:continue
            if key in episodes:raise ValueError('Duplicate prepared physical episode')
            entry=selected[key]
            if _native_record_source(original,root)!=entry['source']:
                raise ValueError('Prepared cache source differs from authoritative selection')
            if entry.get('native_metadata_sha256'):
                native=json.loads((root/original['native_source']['provenance_path']).read_text())
                metadata=[f['sha256'] for f in native.get('files',[]) if f.get('path')=='zarr.json']
                if metadata!=[entry['native_metadata_sha256']]:raise ValueError('Native root metadata changed after source selection')
            if 'frames' in entry and original['frames']!=entry['frames']:
                raise ValueError('Prepared source length differs from selected native metadata')
            ep=copy.deepcopy(original);ep['split']=entry['split']
            for name in ('path','rgb_path'):ep[name]=str((root/ep[name]).resolve())
            for name in ('action_provenance','native_source'):
                field='path' if name=='action_provenance' else 'provenance_path'
                if name in ep:ep[name][field]=str((root/ep[name][field]).resolve())
            if not Path(ep['path']).is_file() or not Path(ep['rgb_path']).is_file():
                raise ValueError('Prepared cache is missing')
            if link_caches:
                linked=output/'episodes'/ep['episode_id'];linked.mkdir(parents=True,exist_ok=True)
                def link(original_path,filename):
                    src=Path(original_path).resolve(strict=True);dst=linked/filename
                    if Path(original_path).is_symlink():raise ValueError('Prepared source must be a regular immutable file')
                    if dst.exists():
                        if not os.path.samefile(src,dst):raise ValueError('Prepared hardlink destination collision')
                    else:os.link(src,dst)
                    return str(dst.relative_to(output))
                ep['path']=link(ep['path'],'state.h5');ep['rgb_path']=link(ep['rgb_path'],'rgb.h5')
                ep['native_source']['provenance_path']=link(ep['native_source']['provenance_path'],'native.json')
                if 'action_provenance' in ep:ep['action_provenance']['path']=link(ep['action_provenance']['path'],'alignment.json')
            episodes[key]=ep
    if set(episodes)!=set(selected):raise ValueError('Prepared corpus does not cover selection exactly')
    verifier=ArtifactVerificationMemo()
    for ep in episodes.values():
        for field,digest,fingerprint in [('path','sha256','verified_fingerprint'),('rgb_path','rgb_sha256','rgb_verified_fingerprint')]:
            proof=verifier.verify(output/ep[field],ep[digest],trusted_fingerprint=ep.get(fingerprint))
            if proof is not None:ep[fingerprint]=proof
    moments={}
    for key,ep in sorted(episodes.items()):
        if ep['split']!='train':continue
        emb=ep['embodiment']
        with h5py.File(output/ep['path'],'r') as data:
            pose_valid=np.asarray(data['pose_valid'],bool)
            camera_valid=np.asarray(data['camera_pose_valid'],bool)
            usable=pose_valid & camera_valid[:,ep['cameras'].index('front_1')]
            moments.setdefault('pose/'+emb,StreamingMoments()).update(np.asarray(data['pose'])[usable])
            camera_valid &= usable[:,None]
            refs=[int(np.flatnonzero(camera_valid[:,c])[0]) if camera_valid[:,c].any() else 0 for c in range(camera_valid.shape[1])]
            relative,valid=relative_camera_pose(np.asarray(data['camera_world_pose']),camera_valid,refs)
            moments.setdefault('camera/'+emb,StreamingMoments()).update(relative[valid])
    required={kind+'/'+emb for kind in ('pose','camera') for emb in schemas['pose_schemas']}
    if set(moments)!=required:raise ValueError('Every embodiment requires actual train rows')
    output=Path(destination);output.mkdir(parents=True,exist_ok=True)
    ordered=[episodes[k] for k in sorted(episodes)]
    norm=output/'normalization.json'
    immutable_json(norm,dict(schema='pixel-proprio-normalization-v1',scope='train_only',train_episodes=training_identities(ordered),statistics={k:v.result(.05) for k,v in moments.items()},camera_moment_reference='first usable physical episode frame per measured camera; fixed TRAIN-only statistics'))
    result=dict(schema='pixel-proprio-prepared-v1',episodes=ordered,**schemas,pixel_identity=pixels,normalization=dict(scope='train_only',arrays_normalized=False,path=norm.name,sha256=sha(norm)),preparation=dict(selection_sha256=sha(selection_path),immutable_input_manifests=bindings,rgb_decoded_or_copied=False,immutable_hardlinks=link_caches))
    immutable_json(output/'manifest.json',result)
    return result


def _verify_complete_episode(manifest_path, entry):
    path=Path(manifest_path);root=path.parent;prior=json.loads(path.read_text())
    if len(prior['episodes'])!=1:
        raise ValueError('One physical episode per resumable cache required')
    ep=prior['episodes'][0]
    if (ep['episode_id']!=entry['episode_id'] or ep['embodiment']!=entry['embodiment']
            or _native_record_source(ep,root)!=entry['source']):
        raise ValueError('Existing episode cache identity mismatch')
    for key in ('path','rgb_path'):
        file=root/ep[key];identity=ep.get('prepared_file_identity',{}).get(key)
        if not file.is_file() or file.is_symlink() or not identity:
            raise ValueError('Missing verified prepared file identity')
        stat=file.stat()
        if identity!=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,uid=stat.st_uid) or stat.st_size<1:
            raise ValueError('Prepared file identity changed after verification')
    if 'action_provenance' in ep and sha(root/ep['action_provenance']['path'])!=ep['action_provenance']['sha256']:
        raise ValueError('Prepared action provenance changed')
    return prior


def prepare_shard(selection_path, destination, *, source_cache, image_size=(160,160), cleanup_source=False):
    """Resume completed per-episode caches; remove only marked task-owned copies."""
    selection=json.loads(Path(selection_path).read_text());output=Path(destination);output.mkdir(parents=True,exist_ok=True)
    cache=Path(source_cache).resolve();cache.mkdir(parents=True,exist_ok=True)
    completed=[]
    for entry in selection['episodes']:
        eid=entry['episode_id'];target=output/eid;manifest=target/'manifest.json'
        if manifest.exists():
            _verify_complete_episode(manifest,entry)
        else:
            owned=cache/eid
            if not isinstance(eid,str) or not eid or eid in ('.','..') or Path(eid).name!=eid or owned.parent!=cache:raise ValueError('Unsafe task-owned source path')
            marker=cache/(eid+'.owner.json')
            ownership=dict(episode_id=eid,source=entry['source'],uid=os.getuid(),cache_root=str(cache))
            if marker.exists() and json.loads(marker.read_text())!=ownership:raise ValueError('Source cache ownership mismatch')
            if not marker.exists():immutable_json(marker,ownership)
            prepare(dict(episodes=[entry]),target,source_cache=cache,image_size=image_size,defer_normalization=True)
        _verify_complete_episode(manifest,entry)
        completed.append(str(manifest.resolve()))
        (output/'progress.json').write_text(json.dumps(dict(completed=completed,total=len(selection['episodes'])),indent=2)+'\n')
        if cleanup_source:
            owned=cache/eid;marker=cache/(eid+'.owner.json')
            # Only the new staged source copy is removable, after a complete
            # immutable prepared manifest and exact matching ownership marker.
            if owned.exists():
                if not marker.is_file() or owned.is_symlink() or owned.resolve().parent!=cache:
                    raise ValueError('Refusing unowned/escaped native source cleanup')
                record=json.loads(marker.read_text())
                if record!=dict(episode_id=eid,source=entry['source'],uid=os.getuid(),cache_root=str(cache)) or owned.stat().st_uid!=os.getuid():
                    raise ValueError('Native source cleanup ownership differs')
                shutil.rmtree(owned)
    index=output/'shard-manifests.json'
    if index.exists():
        if json.loads(index.read_text())!=dict(manifests=completed):raise ValueError('Completed shard index differs')
    else:
        immutable_json(index,dict(manifests=completed))
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--source-cache',required=True)
    p.add_argument('--image-size',type=int,nargs=2,default=(160,160))
    p.add_argument('--patch-size',type=int,default=16)
    p.add_argument('--defer-normalization',action='store_true')
    p.add_argument('--shard',action='store_true')
    p.add_argument('--cleanup-source',action='store_true')
    p.add_argument('--merge-manifest',action='append',default=[])
    p.add_argument('--link-caches',action='store_true')
    a=p.parse_args()
    if a.merge_manifest:
        merge_prepared_manifests(a.selection,a.merge_manifest,a.output,link_caches=a.link_caches)
    elif a.shard:
        prepare_shard(a.selection,a.output,source_cache=a.source_cache,image_size=tuple(a.image_size),cleanup_source=a.cleanup_source)
    else:
        prepare(a.selection,a.output,source_cache=a.source_cache,image_size=tuple(a.image_size),patch_size=a.patch_size,defer_normalization=a.defer_normalization)

if __name__=='__main__':
    main()

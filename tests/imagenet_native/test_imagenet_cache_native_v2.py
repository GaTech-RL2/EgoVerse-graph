"""Pure synthetic target-v2 evidence graphs; never actual cache/native proof.

The native helper fixtures are exact published bytes, read only for hash closure.
They are never imported. No corpus, wheel, source archive or ELF payload exists.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.train.joint import imagenet, imagenet_wheel_overlay
from test_imagenet_raw_cpu_chain import digest, write
from test_imagenet_raw_cpu_v5_physical import v5_fixture

FIXTURES=Path(__file__).parent/'fixtures'
CAMPAIGN=Path('/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1')
ROOT=CAMPAIGN/'imagenet-target-cpu-v2'
TRANSFER=CAMPAIGN/'imagenet-transfer-controller-v3'
HELPER_SHA='d5167548d4982c9554a40fdb246f5ba0d301a5f2c1d23efbe61a147205031caf'
WHEEL_SHA='42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca'
PROOF_NAMES=('prior-runtime-proof.json','runtime-current.json','container-preflight.json','container-runtime.json',
    'source-stage-binding.json','transfer-code-readback.json','filesystem-check.json')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def native(q,qsha,job='456',state='RUNNING',capture=False):
    r=q['capture_resources'] if capture else q['resources'];cpus=str(r['cpus']);mem=str(r['memory_mib']//1024)+'G'
    f=dict(JobId=job,JobName=q['capture_name' if capture else 'job_name'],UserId='paphiwetsa3(41003)',
        Account='loaner',Partition='rl2',QOS='loaner-normal',ReqNodeList='gpu21',
        NumNodes='1-1' if state=='PENDING' else '1',NumCPUs=cpus,NumTasks='1',MinMemoryNode=mem,
        TimeLimit=r['time_limit'],Requeue='0',Restarts='0',JobState=state,
        Command=str(ROOT/('capture.sbatch' if capture else 'producer.sbatch')),WorkDir=str(ROOT),
        Comment='imagenet-target-observer:'+qsha+':456' if capture else 'imagenet-target-cpu:'+qsha,
        SubmitTime='2026-10-26T22:59:01' if capture else '2026-10-26T22:59:00',
        ReqTRES='cpu='+cpus+',mem='+mem+',node=1',AllocTRES='(null)' if state=='PENDING' else 'cpu='+cpus+',mem='+mem+',node=1',
        TresPerTask='cpu:'+cpus)
    if state=='PENDING':f.update(Reason='Dependency' if capture else 'JobHeldUser',Dependency='afterany:456(unfulfilled)' if capture else '(null)')
    else:f.update(NodeList='gpu21',StartTime='2026-10-26T23:30:01' if capture else '2026-10-26T23:00:00')
    if state=='COMPLETED':f.update(EndTime='2026-10-26T23:30:00',ExitCode='0:0')
    return ' '.join(k+'='+v for k,v in f.items())+'\n'


class SyntheticCacheV2:
    def __init__(self,tmp):
        rawtmp=tmp/'raw';rawtmp.mkdir();self.raw=v5_fixture(rawtmp)
        self.decoder={'version':'13.0.0','synthetic_fixture_only':True}
        self.identity='a'*64
        self.raw.completion.update(content_identity_sha256=self.identity,decoder_binding_sha256=canonical_sha(self.decoder))
        self.raw.refresh()
        self.root=tmp/'target-evidence';self.root.mkdir();self.original=ROOT
        self.blocked=set(self.raw.blocked)
        self.q=dict(schema='imagenet-target-cpu-request-v2',state='BOUND_TARGET_CPU',root=str(ROOT),output=str(ROOT/'output'),
            request_id='SYNTHETIC-ONLY-target-v2',uid=41003,user='paphiwetsa3',node='gpu21',account='loaner',partition='rl2',qos='loaner-normal',
            job_name='elmo-imagenet-target-cache-v2',capture_name='elmo-native-imagenet-target-cache-v2',
            wrapper_protocol='REQUIRES_REVIEWED_LAMBDA_CPU_WRAPPER_V2',resources=dict(cpus=4,gpus=0,memory_mib=24576,time_limit='04:00:00'),
            capture_resources=dict(cpus=2,gpus=0,memory_mib=1024,time_limit='00:10:00'),
            native={k:'/usr/local/bin/'+k for k in ('sbatch','scontrol','squeue')},counts=dict(train=1281167,val=50000),classes=1000,
            verification_id='verify-SYNTHETIC-0000000000000001',training_admitted=False,
            lambda_runtime=dict(scratch_root='/ephemeral/enroot/paphiwetsa3/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1',
                fa4_root='/workspace/pinned-flash4',temporary_root='/ephemeral/enroot/41003'),
            runtime_content_reference=dict(physical_root='/ephemeral/enroot/paphiwetsa3/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/runtime-preflight-v3/root'),
            runtime_prerequisite=dict(request=dict(sha256='b'*64),files_manifest=dict(sha256='c'*64),completion=dict(sha256='d'*64)),
            native_client=dict(schema='native-elmo-client-closure-input-v1',scontrol='/usr/local/bin/scontrol',squeue='/usr/local/bin/squeue',
                config='/etc/slurm/slurm.conf',munge_socket='/run/munge/munge.socket.2'),
            decoder_interpreter=dict(implementation='cpython',major_minor=[3,12],physical_executable='/usr/bin/python3.12'),
            source_evidence_root=str(self.raw.root),source_reader_root='/SYNTHETIC/source-reader',
            source_completion=dict(path=str(self.raw.root/'completion.json'),sha256=digest(self.raw.root/'completion.json')),
            source_raw_inventory=dict(path=str(self.raw.root/'output/raw-inventory.json'),sha256=self.raw.completion['raw_inventory']['sha256']))
        self.q['data_root']=str(CAMPAIGN/'imagenet-staging-v1'/self.identity/'verified'/self.q['verification_id']/'data')
        source=ROOT/'output/source-cache/source';source_members={}
        for p in (FIXTURES/'native-source-i').iterdir():
            name='scripts/train/joint/native_elmo_ops/'+p.name
            self.put('output/source-cache/source/'+name,p.read_text());source_members[name]=digest(p)
        sman=self.put('inputs/source.files.sha256',''.join(v+'  '+k+'\n' for k,v in sorted(source_members.items())))
        self.q['source']=dict(revision='e9bff72c82618e74175886d16983f3cf90780fb0',root=str(source),manifest=sman,
            archive=dict(path=str(ROOT/'inputs/source.tar.gz'),sha256='e'*64))
        self.q['source']['publication']=self.put('inputs/publication-receipt.json',dict(status='PASS',upstream_verified=True,
            revision=self.q['source']['revision'],archive_sha256='e'*64,manifest_sha256=sman['sha256'],synthetic_fixture_only=True))
        self.blocked.add(self.root/'inputs/source.tar.gz')
        wheel=ROOT/'inputs/pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
        self.q['decoder_wheel']=dict(path=str(wheel),sha256=WHEEL_SHA);self.blocked.add(self.root/wheel.relative_to(ROOT))
        self.put('inputs/decoder-service/imagenet_wheel_overlay.py',Path(imagenet_wheel_overlay.__file__).read_text())
        service_manifest=self.put('inputs/decoder-service/SERVICE_FILES.sha256',HELPER_SHA+'  imagenet_wheel_overlay.py\n')
        publication=self.put('inputs/decoder-service-publication.json',dict(status='PASS',upstream_verified=True,revision='f'*40,manifest_sha256=service_manifest['sha256'],synthetic_fixture_only=True))
        self.q['decoder_service']=dict(root=str(ROOT/'inputs/decoder-service'),manifest=service_manifest,
            helper=dict(path=str(ROOT/'inputs/decoder-service/imagenet_wheel_overlay.py'),sha256=HELPER_SHA),revision='f'*40,publication=publication)
        self.transfer_files={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (FIXTURES/'transfer-v3-files.sha256').read_text().splitlines()}
        self.q['transfer']=dict(root=str(TRANSFER),manifest=dict(path=str(TRANSFER/'FILES.sha256'),sha256=digest(FIXTURES/'transfer-v3-files.sha256')),
            payload=dict(path=str(TRANSFER/'verify_allocated.py'),sha256=self.transfer_files['verify_allocated.py']))
        shard='00000-'+'1'*20;attempt='attempt-SYNTHETIC-0000000000000001'
        self.q['plan']=self.put('inputs/plan/plan.json',dict(schema='imagenet-mac-stream-transfer-plan-v1',content_identity_sha256=self.identity,
            raw_inventory_sha256=self.q['source_raw_inventory']['sha256'],source_completion_sha256=self.q['source_completion']['sha256'],
            total_files=1331167,total_raw_bytes=133116700,shards=[dict(shard_id=shard,files=1331167,bytes=133116700)]))
        self.q['selections']=self.put('inputs/selections.json',dict(schema='imagenet-transfer-selection-v1',plan_sha256=self.q['plan']['sha256'],shards=[dict(shard_id=shard,attempt=attempt)]))
        self.put('request.json',self.q);self.qsha=digest(self.root/'request.json')
        code_names=('guard.py','producer.sbatch','capture.sbatch','submit_once.py','run_allocated.py','observe.py','finalize.py',
            'runtime_prereq.py','native_capsule.py','authenticate_decoder.py','decoder_runtime_bootstrap.py','payload_dispatch.py')
        for name in code_names:
            self.put(name,'# SYNTHETIC NEVER EXECUTED\n')
        static=['request.json',*code_names]+[str(p.relative_to(self.root)) for p in (self.root/'inputs').rglob('*') if p.is_file()]
        entries={name:digest(self.root/name) for name in static}
        entries['inputs/source.tar.gz']='e'*64;entries[str(wheel.relative_to(ROOT))]=WHEEL_SHA
        self.put('FILES.sha256',''.join(v+'  '+k+'\n' for k,v in sorted(entries.items())));self.fsha=digest(self.root/'FILES.sha256')
        self.claim=self.root/'claims'/self.qsha
        self.put('claims/'+self.qsha+'/PREPARED.json',dict(request_sha256=self.qsha,files_manifest_sha256=self.fsha,parent_argv=imagenet._cache_argv(self.q,self.qsha,self.fsha)))
        for kind,job,parent in (('parent','456',None),('capture','457','456')):
            base='claims/'+self.qsha+'/'+kind+'-sbatch'
            self.put(base+'.argv.json',imagenet._cache_argv(self.q,self.qsha,self.fsha,parent=parent));self.put(base+'.stdout',job+'\n');self.put(base+'.exit','0\n')
            self.put('claims/'+self.qsha+('/PARENT_JOB_ID' if kind=='parent' else '/CAPTURE_JOB_ID'),job+'\n')
        for name,capture in (('HELD_NATIVE.txt',False),('CAPTURE_PENDING_NATIVE.txt',True),('RELEASE_HELD_NATIVE.txt',False),('RELEASE_CAPTURE_PENDING_NATIVE.txt',True)):
            self.put('claims/'+self.qsha+'/'+name,native(self.q,self.qsha,job='457' if capture else '456',state='PENDING',capture=capture))
        self.put('claims/'+self.qsha+'/RELEASE_ATTEMPT.json',dict(parent='456',capture='457',request_sha256=self.qsha))
        self.put('claims/'+self.qsha+'/RELEASE_ACCEPTED.json',dict(parent='456',capture='457',stdout='',status='RELEASE_ACCEPTED_NOT_COMPLETION'))
        self.put('request-claims/'+self.q['request_id']+'/REQUEST_SHA256',self.qsha+'\n')
        for name in ('root-review.json','deployment-readback.json'):
            self.put(name,dict(status='PASS',root=str(ROOT),request_sha256=self.qsha,files_manifest_sha256=self.fsha))
        self.running=self.put('output/native-running.txt',native(self.q,self.qsha))
        self.put('host-exit-456.txt','0\n');self.put('capture-running-457.txt',native(self.q,self.qsha,job='457',capture=True))
        queries=[]
        for kind,name,value in (('job','native-456.txt',native(self.q,self.qsha,state='COMPLETED')),('steps','native-456.steps.txt','No job steps in the system\n')):
            queries.append(dict(kind=kind,exit_code=0,argv=['/usr/local/bin/scontrol','show','job','456','-o','-dd'] if kind=='job' else ['/usr/local/bin/scontrol','show','step','-o'],
                stdout=self.put(name,value),stderr=self.put(name+'.stderr','')))
        self.capture=dict(schema='imagenet-cache-native-capture-v1',parent_job_id='456',observer_job_id='457',request_sha256=self.qsha,files_manifest_sha256=self.fsha,
            observer_running=self.b('capture-running-457.txt'),queries=queries)
        common=dict(job_id='456',request_sha256=self.qsha)
        self.put('output/prior-runtime-proof.json',dict(status='PASS_PRIOR_RUNTIME_SCOPE_ONLY',prior_request_sha256='b'*64,prior_files_manifest_sha256='c'*64,
            completion_sha256='d'*64,ImageNet_source_data_model_admission=False,GPU_or_training_admission=False))
        self.put('output/runtime-current.json',dict(common,status='ACTUAL_CURRENT_RUNTIME_PASS_CONTAINER_IMPORT_PENDING',all_regular_files_freshly_hashed=True,
            exact_inventory=True,regular_files=72878,symlinks=10,regular_bytes=17361459887,runtime_mutated=False))
        nsource=str(Path(self.q['lambda_runtime']['scratch_root'])/'source/456/source')
        nstage=self.put('output/source-stage-native.json',dict(schema='joint-node-local-source-v2',status='PASS',job_id='456',node='gpu21',source=nsource,
            archive_sha256='e'*64,manifest_sha256=sman['sha256'],all_manifest_files_verified=True))
        projection=self.put('output/source-stage-projection.json',dict(common,schema='imagenet-source-physical-projection-v1',status='SOURCE_PROJECTION_FRESH_MANIFEST_READBACK_PASS',
            native_stage=nstage,native_source=nsource,projected_source=str(source),source_manifest=sman,files=len(source_members),all_manifest_files_verified=True,
            physical_source=True,all_files_readonly=True,all_members_owned=True,files_and_directories_fsynced=True,source_bytes_changed=False))
        self.put('output/source-stage-binding.json',dict(source=self.q['source'],source_inputs_freshly_hashed=True,source_stage_verified=True,source_stage=nstage,projection=projection))
        self.put('output/transfer-code-readback.json',dict(manifest=self.q['transfer']['manifest'],payload=self.q['transfer']['payload'],files=self.transfer_files))
        self.put('output/filesystem-check.json',dict(status='PASS',runtime_mounts_verified=True,write_fsync_verified=True,unix_socket_verified=True))
        reader={
            'guard.py':'44f02f1364c7e474440e38d2b494ad5c0b90d203407e3617874cb212755f9f16','submit_once.py':'eaae1bd0297f80948e36eaef0196da23dce3db6c4a97af83726047a355d87d5a',
            'finalize.py':'210ff464bb42bf71c32205865e8f2e5072394bfb2173e8544cca111ec8bbad60','memory_policy.py':'a9ef364c6714e35be0a1937fef46cab2325b4f3e221ddfd2186aa133983518d9',
            'wheel_overlay.py':'8aa731d672b1a034cdfd17e9356cc409a36e5ee68fe8c22e479a6eb1fa36f631'}
        revalidation=self.put('output/source-chain-revalidation/receipt.json',dict(schema='imagenet-source-completion-revalidation-v1',status='PASS',
            completion_sha256=self.q['source_completion']['sha256'],source_request_sha256=self.raw.qsha,source_files_sha256=self.raw.fsha,
            reader_files=reader,source_evidence_root=str(self.raw.root),image_corpus_copied=False))
        assembly=self.put('output/assembly.json',dict(schema='imagenet-target-assembly-v1',status='ALL_BYTES_HASHED_BEFORE_ML_IMPORT',data_root=self.q['data_root'],
            images=1331167,raw_bytes=133116700,plan_sha256=self.q['plan']['sha256'],selections_sha256=self.q['selections']['sha256'],shared_inode_assembly=True,data_admitted=False,
            selected_attempts=[dict(shard_id=shard,attempt=attempt,intent_sha256='1'*64,exit_sha256='2'*64)]))
        self.host=dict(common,schema='imagenet-target-host-preparation-v2',status='ALL_BYTES_HASHED_BEFORE_ML_IMPORT',data_root=self.q['data_root'],source=self.q['source'],
            counts=self.q['counts'],classes=1000,source_decoder=self.decoder,raw_content_identity_sha256=self.identity,source_raw_inventory=self.q['source_raw_inventory'],
            source_manifest_sha256=self.q['source_raw_inventory']['sha256'],source_completion=self.q['source_completion'],source_revalidation=revalidation,assembly=assembly,
            native_running=self.running,storage=dict(file_fsync=True,directory_fsync=True,free_bytes_before_assembly=16*1024**3),no_ml_modules_loaded=True,data_admitted=False)
        self.put('output/host-preparation.json',self.host)
        members={name:dict(source='/usr/lib/SYNTHETIC/'+name.replace('/','_'),sha256=str(i+1)*64) for i,name in enumerate(('bin/scontrol','bin/squeue','etc/slurm.conf','lib/ld-linux-x86-64.so.2'))}
        nmanifest=self.put('output/native-client-files.sha256',''.join(v['sha256']+'  '+k+'\n' for k,v in members.items()))
        croot=str(Path(self.q['lambda_runtime']['scratch_root'])/'native-clients/456');logical='/run/elmo-slurm-client';directory=source/'scripts/train/joint/native_elmo_ops'
        stage=self.put('output/native-client-stage.json',dict(schema='native-elmo-client-stage-v1',phase='imagenet_target_cpu',job_id='456',parent_request=self.b('request.json'),
            capsule=croot,logical_container_root=logical,manifest_sha256=nmanifest['sha256'],members=members,container_native_query_certified=False,munge_keys_read_or_copied=False))
        capsule=dict(common,schema='imagenet-target-native-capsule-v1',status='STAGED_CONTAINER_AUTH_PENDING',source_revision=self.q['source']['revision'],logical_root=logical,
            members_verified=True,container_native_query_certified=False,munge_keys_read_or_copied=False,stage=stage,manifest_readback=nmanifest,manifest_sha256=nmanifest['sha256'],
            implementation={name:self.b(str((directory/name).relative_to(ROOT))) for name in ('native_client.py','wrapper_common.py','native_scontrol.sh','native_squeue.sh')},
            exports=dict(ELMO_NATIVE_CLIENT_HOST_ROOT=croot,ELMO_NATIVE_CLIENT_ROOT=logical,ELMO_NATIVE_CLIENT_MANIFEST_SHA256=nmanifest['sha256'],
                ELMO_NATIVE_MOUNTS=','.join((croot+':'+logical+':ro',str(directory/'native_scontrol.sh')+':/usr/local/bin/scontrol:ro',str(directory/'native_squeue.sh')+':/usr/local/bin/squeue:ro','/run/munge:/run/munge:ro'))))
        ns={name:name+':['+str(i+1)+']' for i,name in enumerate(('mnt','pid','user','uts','net'))};self.ns=ns
        self.pre=dict(common,schema='imagenet-target-container-preflight-v1',status='AUTHENTICATED_BEFORE_TORCH',step_id='0',namespaces=ns,zero_gpu=True,
            python=dict(physical_executable='/usr/bin/python3.12',version='3.12.7',implementation='cpython',system='Linux',machine='x86_64'),
            source_revision=self.q['source']['revision'],source_manifest_sha256=sman['sha256'],files_manifest_sha256=self.fsha,source_files=len(source_members),
            runtime_metadata=dict(torch='2.9.1+cu126',pillow='13.0.0'),host_native=self.running,capsule=capsule)
        for key,stem,args,value in (('native_query','container-native-running',['/usr/local/bin/scontrol','show','job','456','-o','-dd'],native(self.q,self.qsha)),
            ('native_queue','container-native-queue',['/usr/local/bin/squeue','--noheader','--jobs','456','-o','%i|%j|%u|%T|%N|%k'],
             '456|'+self.q['job_name']+'|paphiwetsa3|RUNNING|gpu21|imagenet-target-cpu:'+self.qsha+'\n')):
            self.pre[key]=dict(argv=args,exit_code=0,observed_at='2026-10-26T23:05:00+00:00',stdout=self.put('output/'+stem+'.txt',value),stderr=self.put('output/'+stem+'.stderr',''))
        self.put('output/container-preflight.json',self.pre)
        inspector=json.loads((FIXTURES/'pillow-cp312-wheel-members.json').read_text())['inspection'];overlay=ROOT/'output/pillow-overlay'
        install=dict(schema='pillow-private-wheel-install-v1',status='EXACT_RECORD_AND_EXTRACTED_BYTES_PASS',overlay=str(overlay),**inspector,
            source_stat_unchanged=True,all_extracted_files_reopened_sha256=True,files_and_directories_fsynced=True,pip_used=False,network_used=False,historical_environment_mutated=False)
        druntime=dict(schema='imagenet-target-decoder-runtime-v1',pillow_version='13.0.0',python_version='3.12.7',python_implementation='cpython',system='Linux',machine='x86_64',
            executable='/usr/bin/python3.12',pillow_origin=str(overlay/'PIL/__init__.py'),native_decoder_origin=str(overlay/'PIL/_imaging.cpython-312-x86_64-linux-gnu.so'),
            wheel=self.q['decoder_wheel'],installation=install)
        self.auth=dict(common,schema='imagenet-target-container-authentication-v2',status='NATIVE_AND_DECODER_AUTHENTICATED_BEFORE_TORCH',step_id='0',namespaces=ns,
            host_preparation=self.b('output/host-preparation.json'),native_running=self.put('output/container-payload-native-running.txt',native(self.q,self.qsha)),
            decoder_runtime=druntime,torch_not_imported=True,data_admitted=False)
        self.put('output/container-authentication.json',self.auth)
        self.bootstrap=dict(common,schema='imagenet-target-decoder-prebootstrap-v1',status='EXACT_DECODER_ACTIVE_BEFORE_MAINTAINED_TORCH_BOOTSTRAP',step_id='0',namespaces=ns,
            files_manifest_sha256=self.fsha,host_preparation=self.b('output/host-preparation.json'),container_authentication=self.b('output/container-authentication.json'),
            native_running=self.put('output/prebootstrap-native-running.txt',native(self.q,self.qsha)),decoder_runtime=copy.deepcopy(druntime),payload=self.q['transfer']['payload'],
            torch_not_imported=True,base_environment_mutated=False)
        query=dict(argv=['/usr/local/bin/scontrol','show','job','456','-o','-dd'],exit_code=0,
            stdout=self.bootstrap['native_running'],stderr=self.put('output/prebootstrap-native-running.stderr',''),
            observed_at='2026-10-26T23:06:00+00:00',failure=None)
        query_record=self.put('output/prebootstrap-native-running.query.json',query)
        self.bootstrap['native_query']=dict(query,record=query_record)
        self.put('output/decoder-prebootstrap.json',self.bootstrap)
        self.imported=dict(common,schema='imagenet-target-runtime-import-v1',status='CPU_IMPORT_PASS',step_id='0',namespaces=ns,
            torch_version='2.9.1+cu126',torch_origin=self.q['lambda_runtime']['fa4_root']+'/torch-runtime/torch/__init__.py',pillow_version='13.0.0',
            pillow_origin=str(overlay/'PIL/__init__.py'),real_executable='/usr/bin/python3.12',python_version='3.12.7',payload=self.q['transfer']['payload'],zero_gpu=True,
            python_implementation='cpython',system='Linux',machine='x86_64',
            preflight=self.b('output/container-preflight.json'),host_preparation=self.b('output/host-preparation.json'),container_authentication=self.b('output/container-authentication.json'),
            decoder_prebootstrap=self.b('output/decoder-prebootstrap.json'))
        self.put('output/container-runtime.json',self.imported)
        self.report=dict(common,schema='imagenet-cache-readback-v2',status='EXTERNAL_COMPLETION_REQUIRED',data_admitted=False,data_root=self.q['data_root'],
            raw_content_identity_sha256=self.identity,source_raw_inventory=self.q['source_raw_inventory'],source_manifest_sha256=self.q['source_raw_inventory']['sha256'],
            source_completion=self.q['source_completion'],source_revalidation=revalidation,counts=self.q['counts'],classes=1000,all_image_bytes_hashed=True,all_images_decoded=True,
            destination_inventory=dict(path=str(ROOT/'output/destination-raw.json'),sha256='6'*64),decoder=self.decoder,assembly=assembly,native_running=self.running,
            source=self.q['source'],storage=self.host['storage'],host_preparation=self.b('output/host-preparation.json'),container_authentication=self.b('output/container-authentication.json'),
            host_completion='EXTERNAL_REQUIRED',native_completion='EXTERNAL_REQUIRED',afterany_completion='EXTERNAL_REQUIRED')
        self.blocked.add(self.root/'output/destination-raw.json')
        self.done=dict(schema='imagenet-cache-readback-completion-v2',status='PASS',request_sha256=self.qsha,files_manifest_sha256=self.fsha,job_id='456',data_root=self.q['data_root'],
            host_exit=self.b('host-exit-456.txt'),native_terminal=self.capture['queries'][0]['stdout'],native_start='2026-10-26T23:00:00',native_end='2026-10-26T23:30:00',training_admitted=False)
        self.training=copy.deepcopy(self.raw.request);self.training['data_cache']=self.q['data_root'];self.refresh()

    def put(self,name,value):write(self.root/name,value);return self.b(name)
    def b(self,name):return dict(path=str(ROOT/name),sha256=digest(self.root/name))
    def actual(self,item):return self.root/Path(item['path']).relative_to(ROOT)
    def change(self,name,mutate):
        value=json.loads((self.root/name).read_text());mutate(value);write(self.root/name,value);self.refresh()
    def refresh(self):
        # Rehash only existing bounded metadata leaves and propagate their
        # bindings. Missing corpus/wheel/archive/ELF payloads stay unopened.
        def update(value):
            if isinstance(value,dict):
                if set(value)=={'path','sha256'} and Path(value['path']).is_relative_to(ROOT):
                    p=self.actual(value)
                    if p.is_file() and p not in self.blocked:value['sha256']=digest(p)
                else:
                    for x in value.values():update(x)
            elif isinstance(value,list):
                for x in value:update(x)
        for _ in range(4):
            for path in sorted((self.root/'output').rglob('*.json')):
                v=json.loads(path.read_text());update(v);write(path,v)
        self.report.update(host_preparation=self.b('output/host-preparation.json'),container_authentication=self.b('output/container-authentication.json'))
        update(self.report);self.put('output/cache-readback.json',self.report)
        update(self.capture);self.put('native-456.capture.json',self.capture)
        self.done.update(cache_report_sha256=digest(self.root/'output/cache-readback.json'),native_capture=self.b('native-456.capture.json'),
            producer_proofs={name:self.b('output/'+name) for name in PROOF_NAMES},host_preparation=self.b('output/host-preparation.json'),
            container_authentication=self.b('output/container-authentication.json'),decoder_prebootstrap=self.b('output/decoder-prebootstrap.json'))
        self.put('completion.json',self.done)
        self.packet=dict(schema='imagenet-cache-native-evidence-v1',request=dict(path=str(self.root/'request.json'),sha256=self.qsha),
            completion=dict(path=str(self.root/'completion.json'),sha256=digest(self.root/'completion.json')),evidence_root=str(self.root))
        self.report_binding=self.b('output/cache-readback.json')


def test_complete_synthetic_target_v2_chain_is_metadata_only(tmp_path):
    f=SyntheticCacheV2(tmp_path)
    assert imagenet.verify_cache_native(f.packet,f.report_binding,f.training)==f.done
    assert not any(p.exists() for p in f.blocked)


@pytest.mark.parametrize('name,field,value',[
    ('host-preparation.json','no_ml_modules_loaded',False),('host-preparation.json','status','PENDING'),
    ('assembly.json','images',10),('source-chain-revalidation/receipt.json','completion_sha256','0'*64),
    ('container-preflight.json','step_id','1'),('container-authentication.json','torch_not_imported',False),
    ('decoder-prebootstrap.json','torch_not_imported',False),('decoder-prebootstrap.json','base_environment_mutated',True),
    ('container-runtime.json','step_id','1'),('container-runtime.json','real_executable','/tmp/python'),
    ('source-stage-projection.json','all_files_readonly',False),('source-stage-native.json','source','/tmp/other'),
    ('runtime-current.json','all_regular_files_freshly_hashed',False),
    ('container-runtime.json','torch_origin','/usr/lib/torch/__init__.py')])
def test_rehashed_phase_damage_rejected(tmp_path,name,field,value):
    f=SyntheticCacheV2(tmp_path);f.change('output/'+name,lambda v:v.__setitem__(field,value))
    with pytest.raises(ValueError):imagenet.verify_cache_native(f.packet,f.report_binding,f.training)


@pytest.mark.parametrize('damage',['native_exit','native_argv','native_stderr','native_identity','queue','namespace','capsule_member','capsule_mounts',
    'cp311','native_origin','wheel_record','wheel_pip','wrong_source','wrong_transfer','missing_prebootstrap'])
def test_rehashed_runtime_or_native_damage_rejected(tmp_path,damage):
    f=SyntheticCacheV2(tmp_path)
    if damage=='native_exit':f.change('output/container-preflight.json',lambda v:v['native_query'].__setitem__('exit_code',1))
    if damage=='native_argv':f.change('output/container-preflight.json',lambda v:v['native_query'].__setitem__('argv',['fake']))
    if damage=='native_stderr':write(f.root/'output/container-native-running.stderr','failure');f.refresh()
    if damage=='native_identity':
        p=f.root/'output/container-native-running.txt';p.write_text(p.read_text().replace('StartTime=2026-10-26T23:00:00','StartTime=2026-10-26T23:01:00'));f.refresh()
    if damage=='queue':write(f.root/'output/container-native-queue.txt','wrong\n');f.refresh()
    if damage=='namespace':f.change('output/container-authentication.json',lambda v:v['namespaces'].__setitem__('mnt','mnt:[999]'))
    if damage=='capsule_member':f.change('output/native-client-stage.json',lambda v:v['members']['bin/scontrol'].__setitem__('sha256','0'*64))
    if damage=='capsule_mounts':f.change('output/container-preflight.json',lambda v:v['capsule']['exports'].__setitem__('ELMO_NATIVE_MOUNTS','/lib:/lib:ro'))
    if damage in ('cp311','native_origin','wheel_record','wheel_pip'):
        def alter(v):
            d=v['decoder_runtime']
            if damage=='cp311':d['python_version']='3.11.14'
            if damage=='native_origin':d['native_decoder_origin']='/usr/lib/PIL/native.so'
            if damage=='wheel_record':d['installation']['records'].pop(next(iter(d['installation']['records'])))
            if damage=='wheel_pip':d['installation']['pip_used']=True
        f.change('output/container-authentication.json',alter)
    if damage=='wrong_source':f.change('output/source-chain-revalidation/receipt.json',lambda v:v.__setitem__('source_request_sha256','0'*64))
    if damage=='wrong_transfer':f.change('output/transfer-code-readback.json',lambda v:v['files'].__setitem__('common.py','0'*64))
    if damage=='missing_prebootstrap':(f.root/'output/decoder-prebootstrap.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):imagenet.verify_cache_native(f.packet,f.report_binding,f.training)


def test_gateway_never_imports_producer_or_opens_payload_elf_or_wheel(tmp_path):
    f=SyntheticCacheV2(tmp_path)
    code='''import importlib.abc,json,pathlib,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name in ("torch","PIL","egomimic","native_capsule","target_wheel_overlay","verify_allocated","wheel_overlay") or name.startswith(("torch.","PIL.","egomimic.")):
   raise AssertionError("Gateway imported producer/ML/native code: "+name)
sys.meta_path.insert(0,Block())
from scripts.train.joint import imagenet,contract
packet,report,training,blocked=json.loads(sys.stdin.read());blocked={pathlib.Path(p) for p in blocked}
real_open=pathlib.Path.open;real_sha=contract.sha
def opened(self,*args,**kwargs):
 assert self not in blocked,"Gateway opened a payload, wheel or ELF"
 return real_open(self,*args,**kwargs)
def hashed(path):
 assert pathlib.Path(path) not in blocked,"Gateway hashed a payload, wheel or ELF"
 return real_sha(path)
pathlib.Path.open=opened;contract.sha=hashed
assert imagenet.verify_cache_native(packet,report,training)["training_admitted"] is False
assert "torch" not in sys.modules and "PIL" not in sys.modules
print("PASS_SYNTHETIC_TARGET_V2_METADATA_ONLY")
'''
    blocked=set(f.blocked)|{Path(f.q['data_root'])/'train/forbidden.JPEG',Path('/run/elmo-slurm-client/lib/ld-linux-x86-64.so.2')}
    source=Path(imagenet.__file__).resolve().parents[3]
    result=subprocess.run([sys.executable,'-c',code],cwd=source,env=dict(os.environ,PYTHONPATH=str(source)),
        input=json.dumps([f.packet,f.report_binding,f.training,[str(p) for p in blocked]]),capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='PASS_SYNTHETIC_TARGET_V2_METADATA_ONLY'


@pytest.mark.parametrize('name',['cache-readback.json.failure.json','destination-raw.json.failure.json'])
def test_failure_marker_vetoes_an_otherwise_rehashed_success(tmp_path,name):
    f=SyntheticCacheV2(tmp_path);write(f.root/'output'/name,dict(status='FAIL',synthetic_fixture_only=True))
    with pytest.raises(ValueError,match='explicit failure'):
        imagenet.verify_cache_native(f.packet,f.report_binding,f.training)


@pytest.mark.parametrize('damage',['missing','failed','failure','argv','time','record','stderr'])
def test_rehashed_prebootstrap_native_query_damage_rejected(tmp_path,damage):
    f=SyntheticCacheV2(tmp_path)
    def mutate(v):
        if damage=='missing':v.pop('native_query');return
        q=v['native_query']
        if damage=='failed':q['exit_code']=1
        if damage=='failure':q['failure']='TimeoutExpired'
        if damage=='argv':q['argv']=['fake']
        if damage=='time':q['observed_at']='2026-10-26T22:00:00+00:00'
        if damage=='record':q['observed_at']='2026-10-26T23:07:00+00:00'
    if damage=='stderr':write(f.root/'output/prebootstrap-native-running.stderr','diagnostic');f.refresh()
    else:f.change('output/decoder-prebootstrap.json',mutate)
    with pytest.raises((ValueError,KeyError)):imagenet.verify_cache_native(f.packet,f.report_binding,f.training)

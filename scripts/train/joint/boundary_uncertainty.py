"""Pure opt-in admission for bounded annotation uncertainty; no ML imports.

This verifies a new data policy rather than relabeling strict-boundary evidence.
The sampler/data candidate and this operational candidate must be reviewed and
source-bound together. Missing census/native proof fails closed.
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import re

POLICY = dict(schema='annotation-boundary-uncertainty-sampling-v1',policy='annotation_boundary_uncertainty',
              boundary_probability=0.5,maximum_radius_seconds=0.25,duration_fraction=0.1,rng_namespace=0x424E4459)
BASE_REVISION='153d4a5484bb61eb113d9ac7a7b7a5cbabbbf920'
BASE_MANIFEST='208ba5fa51d0514fd27d00f0f202cbf73f6e31ab7f3537e26ca55004e4fe3f99'
SAMPLER_FILES={
 'egomimic/rldb/boundary_sampling.py':'1522b1c1d9d9f7c24977d6184bc2aeca125a7dd7aea890069dced7e7384c42f0',
 'egomimic/rldb/pixel_proprio.py':'9b4a40f66ac36883553e9c2c14007167d23cdbcfed4aa0aa5afb5d1d7b83e4b1',
 'egomimic/rldb/pixel_proprio_datamodule.py':'ec3fd8280b30cce469078a3901c07811a664efe2c939d3480ed4e73312be8e2c',
 'egomimic/pl_utils/joint_smoke.py':'8da5518c058afab469b7e89ac8940c7b246e4ddb8cc6252bc1163350a36ada1c',
}
STREAMS = ['image','proprio','action','camera']
NOISER = dict(_target_='egomimic.models.joint_image_proprio.CoupledImageProprioNoising',maximum=1.0,active_streams=STREAMS)
LOSSES = {name:dict(_target_='egomimic.pipeline.stages_joint_state_action.JointLossStage',stream=name,weight=1.0,
                   loss=dict(_target_='egomimic.models.pixel_diffusion.PlainPixelMSE',reduction='mean_per_example'),
                   **({'optional':True} if name=='action' else {})) for name in STREAMS}
OBJECTIVE = dict(schema='boundary-independent-noise-plain-objective-v1',training_conditioning='independent_frame_noise',
                 noising=NOISER,loss_stages=LOSSES,forced_clean_initial_frame=False)
UNCERTAINTY_FIELDS={'boundary_uncertainty','subtask_boundary_verification_policy','original_subtask_annotation_identity_verified','boundary_policy_sha256','boundary_objective_sha256'}
COUNT_KEYS = {'samples','requested_boundary','realized_boundary','fallback','legacy_uniform','outward_samples',
              'start_inward','start_outward','end_inward','end_outward'}
PROOF_FLAGS = ('synchronized_native_streams_verified','original_instruction_retained','validation_sampling_unchanged')


def need(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    def plain(item):
        if isinstance(item,Mapping):return {k:plain(v) for k,v in item.items()}
        if isinstance(item,Sequence) and not isinstance(item,(str,bytes,bytearray)):
            return [plain(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def policy(value):
    if value is None:return None
    need(isinstance(value,Mapping) and dict(value)==POLICY
         and all(type(value[k]) is float for k in ('boundary_probability','maximum_radius_seconds','duration_fraction'))
         and type(value['rng_namespace']) is int,'Exact typed boundary uncertainty policy required')
    return deepcopy(POLICY)


def sampling_config(cfg):
    data=cfg['data'];selected=policy(data.get('boundary_sampling'))
    if selected is None:return None
    need(data['mode']=='continuous' and data['sampling_strategy']=='full_episode_uniform'
         and data['min_frames']==2 and data['max_frames']==64 and data['checkpointable_sampling'] is True
         and data['training_conditioning']=='independent_frame_noise' and data['task']=='visual_state_action'
         and data.get('temporal_gap_schedule') is None and data.get('completion_gap_schedule') is None
         and data.get('subtask_manifest_sha256') and data.get('language_features_sha256')
         and cfg['seed']==42,'Boundary arm requires exact checkpointed independent-noise continuous data')
    stages=cfg['model']['pipeline']['stages'];ids=cfg['model']['pipeline']['stage_ids']
    noising=stages[ids['joint_noising']]
    need(digest(noising)==digest(dict(_target_='egomimic.pipeline.stages_joint_state_action.JointNoisingStage',noising=NOISER)),
         'Boundary arm changed independent timestamp noiser')
    need(all(digest(stages[ids[name+'_loss']])==digest(LOSSES[name]) for name in STREAMS),'Boundary arm changed plain FP32 objective')
    need(cfg['model']['training_behavior']['distributed_noise_seed']==42,'Boundary arm changed distributed model-noise seed')
    return selected


def request_config(request,cfg):
    selected=sampling_config(cfg);contract=request.get('boundary_sampling_contract')
    need(not any(k.startswith('boundary') and k!='boundary_sampling_contract' for k in request),
         'Unexpected unbound boundary request metadata')
    if selected is None:
        need(contract is None,'Baseline request cannot admit boundary contract/metadata')
        return None
    need(isinstance(contract,dict) and set(contract)=={'schema','policy','objective','eligibility'}
         and contract['schema']=='boundary-uncertainty-admission-v1'
         and policy(contract['policy'])==selected and digest(contract['objective'])==digest(OBJECTIVE),
         'Request does not bind exact boundary policy/noiser/plain objective')
    need(request['arm']=='continuous','Boundary uncertainty is a continuous arm')
    need(set(contract['eligibility'])=={'report','producer_request','producer_spec','producer_files','producer_completion','claim','native','host_exit'},'Complete allocated eligibility producer bindings required')
    for binding in contract['eligibility'].values():
        need(isinstance(binding,dict) and set(binding)=={'path','sha256'}
             and isinstance(binding['path'],str) and binding['path'].startswith('/')
             and '..' not in binding['path'].split('/')
             and re.fullmatch('[0-9a-f]{64}',str(binding['sha256'])),
             'Allocated eligibility bindings remain unbound or unsafe')
    return contract


def context(cfg,state,sampler=None):
    selected=sampling_config(cfg)
    sampling=state.get('sampling',{});overrides=state.get('validation_overrides',{})
    if selected is None:
        need(sampling.get('boundary_sampling') is None and 'boundary_sampling' not in overrides,
             'Baseline checkpoint context contains boundary sampling identity')
        return None
    need(policy(sampling.get('boundary_sampling'))==selected
         and sampling.get('training_conditioning')=='independent_frame_noise'
         and 'boundary_sampling' in overrides and overrides['boundary_sampling'] is None
         and overrides.get('training_conditioning')=='mixed_context',
         'Checkpoint context must bind training policy and strict legacy validation')
    if sampler is not None:
        need(sampler.get('settings')==sampling and sampler.get('validation_overrides')==overrides,
             'Checkpoint sampler and data context differ on boundary/validation policy')
    return selected


def strict_validation_samples(samples):
    """Held-out query targets may already be NaN-hidden; inspect metadata only."""
    for sample in samples.values():
        records=sample.get('pixel_metadata',[])
        need(records,'Missing actual strict held-out sample metadata')
        for i,row in enumerate(records):
            unit=row.get('subtask',{});need(unit.get('interval_semantics')=='half_open','Held-out subtask semantics changed')
            need('boundary_sampling' not in row,'Boundary policy leaked into held-out sampling')
            indices=list(row.get('frame_indices',[]))
            for key in ('proprio_frame_indices','action_frame_indices','camera_frame_indices'):
                for group in row.get(key,[]):indices.extend(group if isinstance(group,list) else [group])
            need(indices and all(unit['start_idx']<=v<unit['end_idx'] for v in indices),'Held-out targets crossed strict original bounds')
            need(sample['task_id'][i]==unit['task_id'] and sample['language_instruction'][i]==unit['instruction'],
                 'Held-out language differs from original annotation')


def _counts(counts):
    need(isinstance(counts,dict) and set(counts)==COUNT_KEYS
         and all(type(v) is int and v>=0 for v in counts.values()),'Invalid actual boundary realization counters')
    n,r,b,f,u=(counts[k] for k in ('samples','requested_boundary','realized_boundary','fallback','legacy_uniform'))
    need(n==r+u and r==b+f and counts['outward_samples']<=b
         and counts['start_inward']+counts['start_outward']<=b
         and counts['end_inward']+counts['end_outward']<=b,
         'Requested/realized/fallback/offset counters are inconsistent')


def fit_rows(ranks,*,selected,world,per_rank_batch,embodiments,context_sha256):
    if selected is None:
        for row in ranks:
            need(not (UNCERTAINTY_FIELDS & set(row))
                 and row.get('subtask_boundaries_verified') is True,'Baseline fit cannot admit uncertainty evidence')
        return None
    need(policy(selected)==POLICY and len(ranks)==world,'Boundary fit topology/policy changed')
    proof=ranks[0].get('boundary_uncertainty')
    need(isinstance(proof,dict) and proof.get('schema')=='boundary-uncertainty-smoke-realizations-v1'
         and policy(proof.get('policy'))==selected and all(proof.get(k) is True for k in PROOF_FLAGS)
         and proof.get('neighbor_instruction_equivalence_asserted') is False,
         'Missing distinct checked boundary uncertainty evidence')
    all_counts=proof.get('all_rank_counts')
    need(isinstance(all_counts,list) and len(all_counts)==world,'All actual rank boundary counters required')
    expected=set(embodiments)
    for rank,row in enumerate(ranks):
        need(row.get('rank')==rank and row.get('world_size')==world and row.get('global_step')==50
             and row.get('boundary_uncertainty')==proof and row.get('subtask_boundaries_verified') is False
             and row.get('original_subtask_annotation_identity_verified') is True
             and row.get('subtask_boundary_verification_policy')==POLICY['policy']
             and row.get('data_context_sha256')==context_sha256,
             'Boundary fit policy/context/annotation identity differs across ranks')
        local=all_counts[rank]
        need(set(local)==expected and set(row.get('embodiment_optimizer_batches',{}))==expected,
             'Missing actual robot/human boundary training populations')
        need(all(type(v) is int and v>0 for v in row['embodiment_optimizer_batches'].values())
             and sum(row['embodiment_optimizer_batches'].values())==50,
             'Every rank must optimize both real embodiment populations over50 updates')
        for emb,counters in local.items():
            _counts(counters)
            need(counters['samples']==row['embodiment_optimizer_batches'][emb]*per_rank_batch,
                 'Boundary counts differ from actual optimized embodiment batches')
        need(sum(c['samples'] for c in local.values())==50*per_rank_batch,'Boundary counters do not cover all50 updates')
    totals={emb:{key:sum(rank[emb][key] for rank in all_counts) for key in COUNT_KEYS} for emb in expected}
    need(proof.get('global_counts_by_embodiment')==totals
         and all(v['samples']>0 for v in totals.values())
         and sum(v['outward_samples'] for v in totals.values())>0,
         'Boundary global aggregation/actual outward coverage is missing')
    return deepcopy(proof)


def _diagnostic(unit,clock,segments):
    a,b=unit['start_idx'],unit['end_idx'];segments=[s for s in segments if s[0]<=a<b<=s[1]]
    need(len(segments)==1,'Census annotation crosses ambiguous recording continuity')
    lo,hi=segments[0]
    need('continuity_segment' not in unit or unit['continuity_segment']==[lo,hi],'Unit continuity identity differs from actual recording')
    end=b if b<hi else b-1
    duration=clock[b-1]-clock[a];radius=min(.25,.1*duration)
    relative=[t-clock[0] for t in clock]
    bands=[[i for i in range(lo,hi) if abs(relative[i]-relative[c])<=radius] for c in (a,end)]
    pairs=sum(i<j for i in bands[0] for j in bands[1])
    reason=('fewer_than_two_recorded_start_band_candidates' if len(bands[0])<2 else
            'fewer_than_two_recorded_end_band_candidates' if len(bands[1])<2 else
            'fewer_than_two_distinct_ordered_boundary_pairs' if pairs<2 else None)
    eligible=reason is None
    outward=eligible and any((i<a or j>=b) and i<j for i in bands[0] for j in bands[1])
    return dict(unit_sha256=digest(unit),original_bounds=[a,b],continuity_segment=[lo,hi],duration_basis='recorded_last_minus_first',
                duration_seconds=duration,radius_seconds=radius,center_indices=[a,end],
                band_candidate_counts=[len(b) for b in bands],admissible_ordered_pair_count=pairs,
                eligible=eligible,outward_pair_possible=outward,fallback_reason=reason)


def census_argv(q,request_sha,files_sha):
    root=q['remote_root']
    return ['/usr/local/bin/sbatch','--parsable','--hold','--no-requeue','--account=loaner','--qos=loaner-normal',
        '--partition=rl2','--nodelist=gpu11','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=24G','--gres=none',
        '--time=00:45:00','--job-name='+q['job_name'],'--chdir='+root,
        '--output='+root+'/logs/'+q['job_name']+'-%j.out','--error='+root+'/logs/'+q['job_name']+'-%j.err',
        root+'/producer.sbatch',root,request_sha,files_sha]


def census_native(raw,job,name,state,root,resources):
    pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)',raw)
    need(len(pairs)==len(dict(pairs)),'Duplicate shared census native fields')
    fields=dict(pairs)
    need(fields.get('JobId')==job and fields.get('JobName')==name and fields.get('JobState')==state
         and fields.get('Command')==root+'/producer.sbatch'
         and fields.get('UserId')=='paphiwetsa3(41003)' and fields.get('Account')=='loaner'
         and fields.get('QOS')=='loaner-normal' and fields.get('Partition')=='rl2'
         and fields.get('ReqNodeList')==resources['node'] and fields.get('NumCPUs')=='4'
         and fields.get('NumNodes') in ({'1','1-1'} if state=='PENDING' else {'1'}) and fields.get('NumTasks')=='1'
         and fields.get('MinMemoryNode') in {'24G','24576M','24576'}
         and fields.get('TimeLimit')=='00:45:00' and fields.get('Requeue')=='0' and fields.get('Restarts')=='0',
         'Wrong shared census native identity/CPU resources')
    if state=='PENDING':need(fields.get('Reason')=='JobHeldUser','Exact held census acceptance required')
    if state in {'RUNNING','COMPLETED'}:
        need(fields.get('NodeList')==resources['node'],'Wrong actual shared census node')
        try:start=datetime.fromisoformat(fields['StartTime'])
        except (KeyError,ValueError,TypeError):raise ValueError('Missing authoritative shared census StartTime') from None
    if state=='COMPLETED':
        try:end=datetime.fromisoformat(fields['EndTime'])
        except (KeyError,ValueError,TypeError):raise ValueError('Missing authoritative shared census EndTime') from None
        need(start.tzinfo==end.tzinfo and end>start and fields.get('ExitCode')=='0:0','Shared census host/native terminal proof failed')
    for key in ('ReqTRES','AllocTRES'):
        tres=fields.get(key,'').split(',')
        need(not any('gpu' in value.lower() for value in tres),'Shared census may not allocate a GPU')
        if key=='ReqTRES' or state in {'RUNNING','COMPLETED'}:
            need('cpu=4' in tres and 'node=1' in tres and ('mem=24G' in tres or 'mem=24576M' in tres),'Wrong shared census exact TRES')
    return fields



def census_context_contract(spec):
    """Pin derivation before execution without inventing an actual context."""
    return dict(schema='allocated-boundary-context-derivation-v1',
        method='candidate_PixelProprioDataModule.prepare_context_on_verified_train_dataset',
        datamodule=dict(relative_path='candidate/pixel_proprio_datamodule.py',
            sha256=SAMPLER_FILES['egomimic/rldb/pixel_proprio_datamodule.py']),
        resolved_config=dict(relative_path='inputs/resolved.yaml',
            sha256='c2377e6c095974db9a753cab4a0ee2daffad0ddd0b8ceada9b3d6d6ee721efbb'),
        prepared_inputs=spec['data']['hashes'],boundary_sampling=POLICY,dataset_split='train',projection_only=True,
        actual_context_emitted_only_by_allocated_producer=True,
        future_training_requires_exact_actual_context_fingerprint=True)

def eligibility(request,manifest,index,reader):
    contract=request.get('boundary_sampling_contract')
    if contract is None:return None
    need(contract.get('schema')=='boundary-uncertainty-admission-v1' and policy(contract.get('policy'))==POLICY
         and digest(contract.get('objective'))==digest(OBJECTIVE),'Invalid explicit boundary census admission contract')
    bindings=contract['eligibility'];report=reader(bindings['report']);producer=reader(bindings['producer_request'])
    spec_raw=reader(bindings['producer_spec'],False);spec=json.loads(spec_raw)
    files_raw=reader(bindings['producer_files'],False)
    complete=reader(bindings['producer_completion']);claim=reader(bindings['claim'])
    native=reader(bindings['native'],False);host=reader(bindings['host_exit'],False)
    need(host.strip()=='0','Allocated shared census host failed')
    root=producer.get('remote_root','');output=producer.get('output','')
    spec_sha=bindings['producer_spec']['sha256'];request_sha=bindings['producer_request']['sha256'];files_sha=bindings['producer_files']['sha256']
    files={}
    for line in files_raw.splitlines():
        if not line.strip():continue
        parts=line.split(None,1);need(len(parts)==2,'Malformed shared producer FILES manifest')
        checksum,name=parts;name=name.lstrip('*').removeprefix('./')
        need(name not in files,'Duplicate shared producer FILES entry')
        files[name]=checksum
    need(files.get('request.json')==request_sha and files.get('spec.json')==spec_sha
         and all(files.get(name)==wanted for name,wanted in spec.get('code',{}).items()),
         'Shared producer FILES does not bind exact request/spec/code closure')
    resources=dict(node=request['lambda_runtime']['node'],cpus=4,memory_gib=24,time_limit='00:45:00',gpus=0)
    need(producer.get('schema')=='train-calibration-and-boundary-census-request-v1'
         and producer.get('spec')==bindings['producer_spec']
         and producer.get('binding_state')=='BOUND_READY_FOR_ROOT_REVIEW' and producer.get('root_review')=='PASS'
         and root.startswith('/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/calibration-census-')
         and '..' not in root.split('/') and output==root+'/output'
         and producer.get('node')==resources['node'] and producer.get('uid')==41003 and producer.get('resources')==resources,
         'Shared census request is not actually bound/reviewed on the exact CPU resources')
    need(spec.get('schema')=='train-calibration-and-boundary-census-spec-v1'
         and spec.get('resources')==resources
         and spec.get('source',{}).get('revision')==BASE_REVISION
         and spec['source'].get('manifest_sha256')==BASE_MANIFEST
         and spec.get('data',{}).get('root')==request['data_manifest'].rsplit('/',1)[0]
         and spec.get('data',{}).get('hashes',{}).get('manifest.json')==request['data_manifest_sha256']
         and spec['data']['hashes'].get('subtasks.json')==request['subtask_contract']['subtask_manifest']['sha256']
         and spec.get('code',{}).get('candidate/boundary_sampling.py')==SAMPLER_FILES['egomimic/rldb/boundary_sampling.py'],
         'Shared census specification changed base source/exact sampler/data identity')
    derivation=census_context_contract(spec)
    need(spec.get('context_derivation')==derivation and producer.get('context_derivation')==derivation
         and report.get('context_derivation')==derivation and spec.get('config')==derivation['resolved_config']
         and spec.get('code',{}).get('candidate/pixel_proprio_datamodule.py')==derivation['datamodule']['sha256']
         and files.get('inputs/resolved.yaml')==derivation['resolved_config']['sha256'],
         'Shared census changed exact DataModule/config/input context derivation')
    for item in (derivation['datamodule'],derivation['resolved_config']):
        reader(dict(path=root+'/'+item['relative_path'],sha256=item['sha256']),False)
    expected_name='elmo-calib-census-'+spec_sha[:16]
    command=producer.get('command_binding',{})
    need(command==dict(path=root+'/producer.sbatch',sha256=spec['code'].get('producer.sbatch'))
         and re.fullmatch('[0-9a-f]{64}',str(command['sha256'])) and bool(reader(command,False).strip()),
         'Missing immutable shared census command binding')
    need(report.get('schema')=='allocated-boundary-census-v1' and report.get('status')=='CENSUS_PASS'
         and report.get('producer_request_sha256')==request_sha and report.get('producer_spec_sha256')==spec_sha
         and report.get('source_manifest_sha256')==BASE_MANIFEST
         and report.get('boundary_sampler_sha256')==SAMPLER_FILES['egomimic/rldb/boundary_sampling.py']
         and report.get('data_manifest_sha256')==request['data_manifest_sha256']
         and report.get('subtask_manifest_sha256')==request['subtask_contract']['subtask_manifest']['sha256']
         and policy(report.get('policy'))==POLICY and digest(report.get('objective'))==digest(OBJECTIVE)
         and report.get('zero_optimizer_updates') is True
         and report.get('every_clock_read_from_bound_prepared_artifact') is True
         and report.get('source_stat_unchanged') is True,'Actual base+sampler-bound allocated clock census missing')
    # Whole source manifests legitimately differ after applying the candidate.
    # The eventual staged source must contain the exact reviewed sampler/guard
    # bytes used to define this CPU diagnostic's projected data semantics.
    production_manifest=reader(dict(path=request['source_manifest'],sha256=request['source_manifest_sha256']),False)
    production_files={}
    for line in production_manifest.splitlines():
        if not line.strip():continue
        parts=line.split(None,1);need(len(parts)==2,'Malformed production source manifest')
        checksum,name=parts;name=name.lstrip('*').removeprefix('./')
        need(name not in production_files,'Duplicate production source manifest entry')
        production_files[name]=checksum
    for name,wanted in SAMPLER_FILES.items():
        need(production_files.get(name)==wanted,'Published production sampler differs from reviewed census semantics: '+name)
        reader(dict(path=request['source'].rstrip('/')+'/'+name,sha256=wanted),False)
    job=str(report.get('job_id',''));name=producer.get('job_name','')
    need(job.isdecimal() and name==expected_name
         and claim.get('schema')=='train-calibration-and-boundary-census-claim-v1'
         and claim.get('status')=='HELD_NATIVE_ACCEPTANCE_BOUND'
         and claim.get('producer_request_sha256')==request_sha and claim.get('producer_spec_sha256')==spec_sha
         and claim.get('files_manifest_sha256')==files_sha
         and claim.get('job_id')==job and claim.get('job_name')==name,
         'Shared census terminal proof is not bound to exact accepted request/spec/FILES claim')
    claim_root=root+'/claims/'+spec_sha
    need(bindings['claim']['path']==claim_root+'/claim.json'
         and claim.get('accepted_held_native',{}).get('path')==claim_root+'/HELD_NATIVE.txt'
         and claim.get('submit_argv',{}).get('path')==claim_root+'/SUBMIT_ARGV.json',
         'Shared census claim artifacts escaped immutable spec namespace')
    held=reader(claim['accepted_held_native'],False)
    argv=reader(claim['submit_argv'])
    need(argv==census_argv(producer,request_sha,files_sha),'Shared census submission argv differs from exact held producer request')
    held_fields=census_native(held,job,name,'PENDING',root,resources)
    terminal_fields=census_native(native,job,name,'COMPLETED',root,resources)
    need(complete.get('schema')=='train-calibration-and-boundary-census-v1'
         and complete.get('status')=='CPU_DIAGNOSTICS_PASS_HOST_NATIVE_COMPLETION_REQUIRED'
         and complete.get('job_id')==job and complete.get('node')==resources['node']
         and complete.get('producer_request_sha256')==request_sha and complete.get('producer_spec_sha256')==spec_sha
         and complete.get('files_manifest_sha256')==files_sha
         and complete.get('claim_sha256')==bindings['claim']['sha256']
         and complete.get('held_native_sha256')==claim['accepted_held_native']['sha256']
         and complete.get('proofs',{}).get('boundary-unit-census.json')==bindings['report']['sha256']
         and complete.get('training_or_GPU_certified') is False and complete.get('production_scale_bound') is False,
         'Shared CPU diagnostic completion does not bind the actual census/claim')
    running=reader(dict(path=output+'/native-running.json',sha256=complete['proofs']['native-running.json']))
    need(running.get('job_id')==job and running.get('request_sha256')==request_sha
         and running.get('spec_sha256')==spec_sha and running.get('files_manifest_sha256')==files_sha
         and running.get('claim_sha256')==bindings['claim']['sha256']
         and running.get('held_native_sha256')==claim['accepted_held_native']['sha256'],
         'Shared running allocation proof differs from accepted producer claim')
    running_fields=census_native(running['raw'],job,name,'RUNNING',root,resources)
    need(running_fields['StartTime']==terminal_fields['StartTime'],'Terminal census belongs to another allocation interval')
    state=report.get('data_context',{})
    projection=reader(dict(path=output+'/projected-boundary-data-context.json',
        sha256=complete['proofs']['projected-boundary-data-context.json']))
    need(projection.get('schema')=='allocated-projected-boundary-data-context-v1'
         and projection.get('status')=='ACTUAL_METADATA_CONTEXT_PASS' and projection.get('job_id')==job
         and projection.get('context_derivation')==derivation and projection.get('data_context')==state
         and projection.get('data_context_sha256')==report.get('data_context_sha256')
         and projection.get('actual_manifest_read') is True and projection.get('actual_candidate_prepare_context_called') is True
         and projection.get('future_model_or_checkpoint_context_claimed') is False,
         'Allocated actual DataModule context projection is missing or changed')
    need(re.fullmatch('[0-9a-f]{64}',str(report.get('data_context_sha256')))
         and digest(state)==report['data_context_sha256']
         and state.get('manifest_sha256')==request['data_manifest_sha256'] and state.get('manifest')==manifest
         and policy(state.get('sampling',{}).get('boundary_sampling'))==POLICY
         and state.get('sampling',{}).get('training_conditioning')=='independent_frame_noise'
         and 'boundary_sampling' in state.get('validation_overrides',{})
         and state['validation_overrides']['boundary_sampling'] is None
         and state['validation_overrides'].get('training_conditioning')=='mixed_context',
         'Census data context does not bind exact training/strict-heldout policies')
    physical={(e['embodiment'],e['episode_id']):e for e in manifest['episodes'] if e['split']=='train'}
    subtasks={(e['embodiment'],e['episode_id']):e['subtasks'] for e in index['episodes'] if e['split']=='train'}
    rows=report.get('episodes',[]);need(len(rows)==len(physical),'Census must cover every training physical episode only')
    seen=set();totals={emb:dict(units=0,eligible_units=0,fallback_units=0,excluded_short_units=0,outward_possible_units=0) for emb,_ in physical}
    anchor_totals={emb:dict(all=0,eligible=0,outward_possible=0) for emb,_ in physical}
    for row in rows:
        key=row['embodiment'],row['episode_id'];need(key in physical and key not in seen,'Census duplicated/changed physical episode')
        seen.add(key);ep=physical[key];clock=row['clock_seconds']
        need(row.get('continuity_segments_explicit')==('continuity_segments' in ep)
             and row.get('rgb_sha256')==ep['rgb_sha256'] and len(clock)==ep['frames']
             and all(type(t) in (int,float) and math.isfinite(t) for t in clock)
             and all(x<y for x,y in zip(clock,clock[1:])),'Census clock is not bound native increasing physical time')
        units=subtasks[key];eligible_units=[u for u in units if u['end_idx']-u['start_idx']>=2]
        expected=[_diagnostic(u,clock,ep.get('continuity_segments',[[0,len(clock)]])) for u in eligible_units]
        need(row.get('units')==expected,'Census eligibility diagnostics differ from actual clocks/original annotations')
        excluded=[digest(u) for u in units if u['end_idx']-u['start_idx']<2]
        need(row.get('excluded_short_unit_sha256')==excluded,'Census changed short-unit exclusion')
        counts=totals[key[0]];counts['units']+=len(expected);counts['excluded_short_units']+=len(excluded)
        counts['eligible_units']+=sum(x['eligible'] for x in expected)
        counts['fallback_units']+=sum(not x['eligible'] for x in expected)
        counts['outward_possible_units']+=sum(x['outward_pair_possible'] for x in expected)
        for unit,diagnostic in zip(eligible_units,expected):
            length=unit['end_idx']-unit['start_idx'];mass=anchor_totals[key[0]]
            mass['all']+=length;mass['eligible']+=length*diagnostic['eligible']
            mass['outward_possible']+=length*diagnostic['outward_pair_possible']
    need(seen==set(physical) and report.get('totals_by_embodiment')==totals
         and report.get('anchor_weighted_totals_by_embodiment')==anchor_totals
         and sum(c['outward_possible_units'] for c in totals.values())>0,
         'Census aggregation/actual outward eligibility missing')
    return report


def completion(request,smoke,result,preflight):
    selected=request.get('boundary_sampling_contract')
    need(selected==smoke.get('boundary_sampling_contract'),'Full boundary contract differs from completed smoke')
    if selected is None:
        need(not (UNCERTAINTY_FIELDS & set(result)) and result.get('subtask_boundaries_verified') is True,
             'Baseline completion cannot admit uncertainty evidence')
        fit_rows(result['ranks'],selected=None,world=request['world_size'],per_rank_batch=request['per_rank_batch'],
                 embodiments=request['interfaces']['pose_dims'],context_sha256=result.get('data_context_sha256'))
        return None
    need(policy(selected.get('policy'))==POLICY and digest(selected.get('objective'))==digest(OBJECTIVE),
         'Full request changed boundary policy/plain objective')
    context_sha=preflight['provenance']['data_context_sha256']
    need(result.get('data_context_sha256')==context_sha and result.get('subtask_boundaries_verified') is False
         and result.get('original_subtask_annotation_identity_verified') is True
         and result.get('subtask_boundary_verification_policy')==POLICY['policy'],
         'Completion must distinguish uncertainty from strict original-boundary proof')
    proof=fit_rows(result['ranks'],selected=POLICY,world=request['world_size'],per_rank_batch=request['per_rank_batch'],
                   embodiments=request['interfaces']['pose_dims'],context_sha256=context_sha)
    need(result.get('boundary_uncertainty')==proof
         and result.get('boundary_objective_sha256')==digest(OBJECTIVE)
         and result.get('boundary_policy_sha256')==digest(POLICY),
         'Postflight does not bind checked policy/objective/rank evidence')
    expected=dict(boundary_policy_sha256=digest(POLICY),boundary_objective_sha256=digest(OBJECTIVE),
                  boundary_eligibility_report_sha256=selected['eligibility']['report']['sha256'])
    need(all(preflight['provenance'].get(k)==v for k,v in expected.items()),'Smoke preflight lacks exact boundary census/policy/objective binding')
    return proof

"""NPU warp自主Pack/Collect候选：直接对照已验收Sparse Pack，CPU仅作冻结对手。"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import sparse_pack as engine,warp_pipeline_sources as source

g=engine.g;r=engine.r;simd=engine.simd;retained=engine.retained;coverage=engine.coverage
digest=retained.corpus.digest
SPARSE_OWN=engine.OWN
SPARSE_PASS=engine.PASS
SPARSE_MISS=engine.MISS
sparse_decision=engine.decision
OWN=('standalone/warp_pipeline.py','standalone/warp_pipeline_sources.py')+SPARSE_OWN
PASS='actual_warp_pipeline_candidate_retained'
MISS='actual_warp_pipeline_correct_but_not_retained'
CHECKED='actual_warp_pipeline_all_scale_coverage_checked'
command_for=engine.command_for
adversarial_documents=engine.adversarial_documents


def actual_atomic_headers(toolkit):
    result=engine.actual_atomic_headers(toolkit)
    # Shuffle declarations can be in another header of the same SIMT include
    # directory. Do not mistake a narrow filename scan for an unsupported API.
    headers=set(Path(result['path']).parent.rglob('*.h'))
    headers.update(Path(name) for name in result['files'])
    declarations={}
    for name in ('atomicAdd','atomicMin','__shfl_up','asc_shfl'):
        rows=[]
        for path in sorted(headers):
            for line_number,line in enumerate(path.read_text(encoding='utf-8',errors='replace').splitlines(),1):
                if re.search(r'\b'+re.escape(name)+r'\s*\(',line):
                    rows.append({'path':str(path),'line':line_number,'text':line.strip()})
                    result['files'][str(path.resolve())]=digest(path)
        declarations[name]=rows
    result['warp_declarations']=declarations
    result['warp_declarations_not_textually_located']=[name for name,rows in declarations.items() if not rows]
    result['scope']+=' Header text is diagnostic: compiler intrinsics may not have searchable declarations. Actual950 compilation must resolve UB atomicAdd/atomicMin and warp shuffle types/overloads; missing text is not evidence of unsupported API.'
    return result


def decision(rows,n):
    result=sparse_decision(rows,n)
    result['baseline']='accepted_sparse_pack_not_earlier_bitmap_or_pre_bitmap'
    return result


def reused_candidate(a,proofs):
    path=a.candidate_from/'report.json';old=json.loads(path.read_text(encoding='utf-8'))
    r.require(old.get('status') in (PASS,MISS),'只复用已完成正确性和六轮测量的候选')
    r.require(old['source_sha256']==g.checks.sources(OWN+retained.OWN+retained.HOST),'跨规模候选源码改变')
    r.require(decision(old['per_round'],old['docs'])==old['decision'],'复用候选六轮判定不能重算')
    r.require(old['status']==(PASS if old['decision']['retain_candidate'] else MISS),'复用候选状态与门禁不符')
    simd.close.scale.unchanged(old['previous_sha256'],'候选原依赖改变');proofs.update(old['previous_sha256'])
    simd.close.scale.unchanged(old['candidate_sha256'],'已安装候选改变');proofs.update(old['candidate_sha256'])
    proofs[str(path.resolve())]=digest(path)
    candidate=old['candidate']
    r.require(Path(candidate['vendor']).resolve()==(a.vendors/source.VENDOR).resolve(),'复用候选vendor路径不同')
    expected={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in source.candidate_sources(a.original).items()}
    r.require(candidate['metadata']['source_sha256']==expected,'复用候选生成源码不一致')
    r.require(old['affected_checks']['raw_selector']['calls']==54 and
        old['affected_checks']['raw_selector']['status']=='actual_simd_selector_guards_and_strict_bit_rankings_passed' and
        old['affected_checks']['baseline']['output_sha256']==old['affected_checks']['candidate']['output_sha256'],'复用候选缺少受影响检查')
    return old


def build_install(a):
    project=a.output/'project';vendor=a.vendors/source.VENDOR
    r.require(not vendor.exists(),'独立候选vendor已存在，停止覆盖')
    metadata=source.prepare(a.original,project);logs=a.output/'compiler_logs';logs.mkdir()
    env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    g.execute(['bash','build.sh'],a.output/'build.log',env,project,timeout=1800)
    old=simd.t.resource.cache(a.original/'build_out/CMakeCache.txt');new=simd.t.resource.cache(project/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==source.VENDOR and {k:v for k,v in old.items() if k!='vendor_name'}==
        {k:v for k,v in new.items() if k!='vendor_name'},'实际950目标或构建配置改变')
    packages=list((project/'build_out').glob('*.run'));r.require(len(packages)==1,'独立安装包不唯一')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (a.output/'package_list.log').write_bytes(listing.stdout)
    r.require(listing.returncode==0 and set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode(errors='replace')))=={source.VENDOR},'安装包不是独立候选')
    g.execute(['bash',str(packages[0]),'--install-path='+str(a.vendors.parent)],a.output/'install.log',timeout=300)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,args in simd.fused.PROTOTYPES.items():
        found=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',args),'实际候选ABI改变：'+name)
    for name,raw in source.candidate_sources(a.original).items():r.require((project/name).read_bytes()==raw,'候选源码改变：'+name)
    objects=simd.fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    compiler_logs,resources=simd.compiler_resources(a.output)
    return {'vendor':str(vendor),'project':str(project),'metadata':metadata,'device_objects':objects,
        'compiler_logs':compiler_logs,'simt_resources':resources,'missing_resource_fields':'UNKNOWN; no occupancy or spill inference'}


def baseline_evidence(a):
    sweep_path=a.completed/'report.json'
    sweep=json.loads(sweep_path.read_text(encoding='utf-8'))
    r.require(sweep.get('status')=='actual_sparse_pack_all_scale_coverage_checked','需要已完成的Sparse Pack七档报告')
    matches=[x for x in sweep['tiers'] if x['docs']==a.docs]
    r.require(len(matches)==1,'该规模没有Sparse Pack验收证据')
    tier=matches[0];point_path=(a.completed/f'N{a.docs}'/'report.json').resolve()
    r.require(Path(tier['report']).resolve()==point_path and tier['report_sha256']==digest(point_path),'Sparse Pack分档报告路径或哈希改变')
    point=json.loads(point_path.read_text(encoding='utf-8'))
    r.require(point.get('status') in (SPARSE_PASS,SPARSE_MISS) and point['docs']==a.docs,'Sparse Pack分档未完成')
    r.require(point['source_sha256']==g.checks.sources(SPARSE_OWN+retained.OWN+retained.HOST),'Sparse Pack被测源码或工具改变')
    r.require(sparse_decision(point['per_round'],a.docs)==point['decision'],'Sparse Pack六轮判定不能重算')
    r.require(point['status']==(SPARSE_PASS if point['decision']['retain_candidate'] else SPARSE_MISS) and
        tier['status']==point['status'] and tier['decision']==point['decision'],'Sparse Pack总报告与分档判定不一致')
    candidate=point['candidate']
    expected={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in engine.source.candidate_sources(a.original).items()}
    r.require(candidate['metadata']['source_sha256']==expected and candidate['metadata']['vendor_name']==engine.source.VENDOR,
        '基线生成源码必须是已验收的Sparse Pack')
    trusted_vendor=Path(candidate['vendor']).resolve(strict=True)
    r.require(trusted_vendor==(a.vendors/engine.source.VENDOR).resolve(strict=True),'Sparse Pack安装目录改变')
    proofs={str(sweep_path.resolve()):digest(sweep_path),str(point_path):digest(point_path)}
    simd.close.scale.unchanged(point['previous_sha256'],'Sparse Pack前序证据改变');proofs.update(point['previous_sha256'])
    simd.close.scale.unchanged(point['candidate_sha256'],'Sparse Pack基线安装或工程改变');proofs.update(point['candidate_sha256'])
    links=point['per_round'];r.require(len(links)==6,'Sparse Pack基线必须有六轮')
    selected={};npu_runner=None;baseline_vendors=None
    for row in links:
        for role in ('candidate',)+coverage.cpu_roles(a.docs):
            link=row[role];path=Path(link['directory'])/'report.json'
            r.require(digest(path)==link['report_sha256'],'Sparse Pack逐轮报告改变')
            proofs[str(path.resolve())]=digest(path)
            data=json.loads(path.read_text(encoding='utf-8'));command=data['command'];binary=Path(command[0]).resolve()
            r.require(data['summary']==link['summary'] and digest(binary)==data['runner_sha256'],'Sparse Pack逐轮摘要或二进制改变')
            r.require(str(binary) in proofs and proofs[str(binary)]==data['runner_sha256'],'逐轮运行器未纳入既有证据链')
            if role=='candidate':
                r.require(len(command)==14 and command[5]==str(point['device']) and command[7:12]==['10','8','24','5','30'],'NPU基线命令合同改变')
                if npu_runner is None:npu_runner=binary;baseline_vendors=Path(command[6]).resolve()
                r.require(binary==npu_runner and Path(command[6]).resolve()==baseline_vendors,'NPU基线运行器或vendor不一致')
                r.require((baseline_vendors/'bm25_fused_score_topk').resolve(strict=True)==trusted_vendor,
                    'NPU基线运行别名未指向已验收的Sparse Pack安装目录')
            else:
                args=['1','10','1','8','24','5','30'] if role.endswith('_single') else ['32','10','32','8','24','5','30']
                r.require(len(command)==12 and command[5:]==args,'CPU线程或测量合同改变')
            selected.setdefault('npu' if role=='candidate' else role,data)
    inputs=Path(selected['npu']['command'][1]).parent;description,index=retained.corpus.validate(inputs)
    r.require(description==point['workload'] and description['docs']==a.docs and description['queries']==8 and
        description['query_length']==3 and description['k']==10,'基线工作量改变')
    for data in selected.values():
        command=data['command']
        r.require(Path(command[1]).resolve()==(inputs/'index').resolve() and Path(command[2]).resolve()==(inputs/'A.txt').resolve() and
            Path(command[3]).resolve()==(inputs/'B.txt').resolve(),'基线CPU/NPU输入路径不同')
    proofs[str((inputs/'preparation_report.json').resolve())]=digest(inputs/'preparation_report.json')
    return point,selected,inputs,description,index,proofs


def run(a):
    r.require(sys.platform=='linux' and not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'需要A5 Linux和独立新结果目录')
    r.require(all(not a.output.is_relative_to(p) and not p.is_relative_to(a.output) for p in (a.completed,a.original,a.vendors)),'结果目录不能覆盖已有证据')
    old,prior,inputs,description,index,proofs=baseline_evidence(a)
    runner=Path(prior['npu']['command'][0]);baseline_vendors=Path(prior['npu']['command'][6]);device=old['device']
    disk_needed=6*(2+len(coverage.cpu_roles(a.docs)))*4*a.docs*8*4+1024**3
    r.require(disk_needed<shutil.disk_usage(a.output.parent).free,'磁盘预算不足')
    plan=retained.layout(a.docs,index['vocab_size'],index['posting_count'])
    r.require(plan['cpu_peak_known_bytes']+512*1024**2<retained.host_available_bytes()*4//5,'主机内存预算不足')
    own=g.checks.sources(OWN+retained.OWN+retained.HOST);a.output.mkdir(parents=True)
    report={'status':'running','scope':'NPU-only warp Pack/Collect pipeline derived from accepted Sparse Pack; CPU binaries and settings are frozen opponents only.',
        'source_sha256':own,'previous_sha256':proofs,'device':device,'docs':a.docs,'workload':description,
        'memory_plan':plan,'disk_budget_bytes':disk_needed,'per_round':[],
        'predeclared_retention_rule':'A/B:6/6 faster than paired accepted SPARSE PACK NPU,>=5% median reduction,lower median p95; pass all declared CPU configurations.',
        'baseline_sparse_pack_report':str(a.completed/f'N{a.docs}'/'report.json')}
    def save():r.write_json(a.output/'report.json',report)
    def check():
        simd.close.scale.unchanged(proofs,'Sparse Pack基线、前序证据或输入改变')
        r.require(g.checks.sources(OWN+retained.OWN+retained.HOST)==own,'候选源码或工具在运行中改变')
        if 'candidate_sha256' in report:simd.close.scale.unchanged(report['candidate_sha256'],'候选安装或工程改变')
    save()
    try:
        atomic=actual_atomic_headers(a.toolkit);report['actual_atomic_header']=atomic;proofs[atomic['path']]=atomic['sha256'];proofs.update(atomic['files'])
        cpuinfo=Path('/proc/cpuinfo').read_bytes();(a.output/'cpu_info.txt').write_bytes(cpuinfo)
        report['host_context']={'cpu_info_sha256':digest(a.output/'cpu_info.txt'),'allowed_cpu_affinity':sorted(os.sched_getaffinity(0)),
            'openmp_environment':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','GOMP_'))}}
        report['management_before']=retained.accepted.prior.management_snapshot(a.output,'warp_pipeline_before');save()
        g.execute([runner,'--memory',device,a.output/'device_memory.json'],a.output/'memory.log',g.environment(baseline_vendors))
        memory=json.loads((a.output/'device_memory.json').read_text(encoding='utf-8'));report['device_memory_probe']=memory;save()
        r.require(plan['known_device_bytes_upper_bound']+plan['debug_score_bytes']+1024**3<=memory['free']*4//5,'实际设备内存预算不足')
        reused=reused_candidate(a,proofs) if a.candidate_from else None
        report['candidate']=reused['candidate'] if reused else build_install(a)
        if reused:report['candidate_reused_from']=str(a.candidate_from/'report.json')
        save();check()
        vendors=g.aliases(a.output/'runtime/vendors',a.vendors/source.VENDOR,baseline_vendors/'bm25_topk_merge')
        project=Path(report['candidate']['project']);files=[p for p in (a.vendors/source.VENDOR).rglob('*') if p.is_file() and p.suffix in ('.so','.o','.json','.h')]
        files += [project/name for name in source.candidate_sources(a.original)]
        files += [project/name for name in ('bm25_project.json','CMakePresets.json','op_kernel/CMakeLists.txt','build.sh')]
        report['candidate_sha256']={str(p.resolve()):digest(p) for p in files};save();check()
        if reused:
            report['affected_checks']=reused['affected_checks'];report['affected_checks_reused_from']=str(a.candidate_from/'report.json');save();check()
        else:
            env=g.environment(vendors);selector=ROOT/'results/fused_simd_close2/selector_host_build/bm25_fused_simd_selector_check'
            r.require(str(selector.resolve()) in proofs,'54项原始分数检查器未被既有证据锁定')
            g.execute([selector,vendors,device,a.output/'selector_checks'],a.output/'selector.log',env)
            raw=json.loads((a.output/'selector_checks/report.json').read_text(encoding='utf-8'))
            r.require(raw['status']=='actual_simd_selector_guards_and_strict_bit_rankings_passed' and raw['calls']==54,'原始分数、同分或保护区检查失败')
            report['affected_checks']={'raw_selector':raw};save();check()
            fixture=a.output/'adversarial_input';original_docs=a.output/'adversarial_tokens.jsonl'
            with original_docs.open('w',encoding='utf-8',newline='\n') as f:
                for row in adversarial_documents():f.write(json.dumps(row)+'\n')
            cases={'A':[[0,0,7],[1,7,7],[2,7,7],[3,7,7]]*2,'B':[[7,7,7],[0,2,0],[2,3,2],[1,3,1]]*2}
            retained.corpus.prepare(fixture,8193,8,950,cases,corpus=original_docs)
            desc,idx=retained.corpus.validate(fixture)
            for label,vendor in (('baseline',baseline_vendors),('candidate',vendors)):
                destination=a.output/('affected_'+label)
                command=[runner,fixture/'index',fixture/'A.txt',fixture/'B.txt',destination,device,vendor,10,8,24,1,10,
                    fixture/'A/expected_scores.bin',fixture/'B/expected_scores.bin']
                check();g.execute(command,a.output/('affected_'+label+'.log'),g.environment(vendor))
                report['affected_checks'][label]=retained.verify_run(destination,fixture,desc,idx,'npu',device,1,10);save();check()
            r.require(report['affected_checks']['baseline']['output_sha256']==report['affected_checks']['candidate']['output_sha256'],'8193边界用例完整输出位不同')
        def measure(role,path):
            path.mkdir(parents=True);out=path/'actual';npu=role in ('baseline','candidate')
            previous=prior['npu'] if npu else prior[role]
            vendor=(vendors if role=='candidate' else baseline_vendors) if npu else None
            command=command_for(previous,out,vendor);check()
            g.execute(command,path/'run.log',g.environment(vendor) if npu else os.environ.copy(),timeout=3600)
            result=coverage.verify_single_cpu(out,inputs,description,index,role) if role.endswith('_single') else retained.verify_run(out,inputs,description,index,'npu' if npu else role,device)
            check();result.update(command=command,runner_sha256=digest(Path(command[0])),log_sha256=digest(path/'run.log'))
            r.write_json(path/'report.json',result)
            print(f'N={a.docs} {role} 完整 Search：'+json.dumps(result['summary'],ensure_ascii=False),flush=True)
            return {'directory':str(path),'summary':result['summary'],'report_sha256':digest(path/'report.json')}
        for i in range(6):
            retained.corpus.validate(inputs);order=coverage.process_order(a.docs,i);row={'round':i+1,'process_order':list(order)}
            for role in order:row[role]=measure(role,a.output/f'round{i+1}'/role)
            for checkpoint in g.checks.CHECKPOINTS:
                left=Path(row['baseline']['directory'])/'actual'/checkpoint;right=Path(row['candidate']['directory'])/'actual'/checkpoint
                for file in left.iterdir():
                    if file.suffix=='.bin':r.require(retained.same_stream(file,right/file.name),'完整分数或树输出位不同')
            report['per_round'].append(row);save()
        retained.corpus.validate(inputs);report['decision']=decision(report['per_round'],a.docs)
        report['management_after']=retained.accepted.prior.management_snapshot(a.output,'warp_pipeline_after')
        report['status']=PASS if report['decision']['retain_candidate'] else MISS;check();save()
        print('NPU warp流水六轮判定：'+json.dumps(report['decision'],ensure_ascii=False),flush=True)
        return 0 if report['status']==PASS else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save()
        logs=list(a.output.rglob('*.log'));recent=max(logs,key=lambda p:p.stat().st_mtime) if logs else None
        if recent:print('最近阶段日志 '+str(recent)+'\n'+recent.read_text(encoding='utf-8',errors='replace')[-24000:],flush=True)
        raise


def sweep(a):
    sizes=coverage.SIZES+(10000000,)
    r.require(sys.platform=='linux' and not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'需要A5 Linux和新结果根目录')
    need=sum(6*(2+len(coverage.cpu_roles(n)))*4*n*8*4 for n in sizes)+2*1024**3
    r.require(need<shutil.disk_usage(a.output.parent).free,'七档不可变输出的磁盘预算不足')
    a.output.mkdir(parents=True)
    report={'status':'running','baseline':'accepted_sparse_pack','planned_sizes':list(sizes),'disk_budget_bytes':need,'tiers':[],
        'scope':'NPU-only warp pipeline; fresh paired full Search at every scale; CPU unchanged. Exit0 means coverage,not all scales winning.'}
    def save():r.write_json(a.output/'report.json',report)
    save();candidate_from=a.candidate_from
    try:
        for n in sizes:
            point=SimpleNamespace(**vars(a));point.docs=n;point.output=a.output/f'N{n}';point.candidate_from=candidate_from
            code=run(point);r.require(code in (0,2),'该规模未完成正确性及测量')
            path=point.output/'report.json';data=json.loads(path.read_text(encoding='utf-8'))
            report['tiers'].append({'docs':n,'status':data['status'],'report':str(path),'report_sha256':digest(path),'decision':data['decision']})
            if candidate_from is None:candidate_from=point.output
            save()
        report['status']=CHECKED
        report['stable_cpu_winning_docs']=[x['docs'] for x in report['tiers'] if x['decision']['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']]
        report['retained_candidate_docs']=[x['docs'] for x in report['tiers'] if x['decision']['retain_candidate']]
        save();print('NPU warp流水全规模覆盖：'+json.dumps(report,ensure_ascii=False),flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,path in {'completed':'results/sparse_pack_scales1','original':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'))
    p.add_argument('--docs',type=int,help='只测一个已验收规模；默认完成全部七档')
    p.add_argument('--candidate-from',type=Path,help='复用本脚本已完成候选，不重编译或重做旧边界检查')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:
        if a.docs is not None:
            r.require(a.docs in coverage.SIZES+(10000000,),'只能测已验收的七个规模');return run(a)
        return sweep(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；保留Sparse Pack基线，不生成替代设备结果。',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())

"""稀疏 Pack 独立候选的 A5 验收；复用冻结基线、CPU 和全部正确性检查。"""
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
from standalone import large_search as retained,score_bitmap as bitmap,sparse_pack_sources as source,score_bitmap_scales as coverage

g=retained.g;r=retained.r;simd=retained.simd;digest=retained.corpus.digest
OWN=('standalone/sparse_pack.py','standalone/sparse_pack_sources.py','standalone/score_bitmap_scales.py')+bitmap.OWN
PASS='actual_sparse_pack_candidate_retained'
MISS='actual_sparse_pack_correct_but_not_retained'
actual_atomic_headers=bitmap.actual_atomic_headers
command_for=bitmap.command_for
adversarial_documents=bitmap.adversarial_documents


def decision(rows,n):
    result=bitmap.decision(rows)
    result['versus_all_declared_cpu_configs']=coverage.cpu_gate(rows,coverage.cpu_roles(n))
    result['retain_candidate']=result['retain_candidate'] and result['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']
    result['baseline']='accepted_score_bitmap_not_pre_bitmap_large'
    return result


def baseline_evidence(a):
    # Validate the accepted bitmap source, installed objects, old report chain and
    # CPU/host binaries without changing any previously tested source bytes.
    old,large,prior,proofs=coverage.accepted_evidence(a)
    if a.docs==old['docs']:
        tier=old;links=old['per_round'];evidence_path=a.accepted/'report.json'
    else:
        evidence_path=a.scales/'report.json'
        sweep=json.loads(evidence_path.read_text(encoding='utf-8'))
        r.require(sweep.get('status')==coverage.CHECKED,'需要已完成的位图版中小规模报告')
        r.require(sweep['source_sha256']==g.checks.sources(coverage.OWN),'位图版覆盖工具或被测源码改变')
        simd.close.scale.unchanged(sweep['previous_sha256'],'位图覆盖依赖证据改变');proofs.update(sweep['previous_sha256'])
        reused=sweep['reused_10m']
        r.require(Path(reused['report']).resolve()==(a.accepted/'report.json').resolve() and
            reused['report_sha256']==digest(a.accepted/'report.json'),'覆盖报告未关联同一已验收位图版')
        matches=[t for t in sweep['tiers'] if t['docs']==a.docs and t.get('status')=='six_rounds_checked']
        r.require(len(matches)==1,'该规模没有已完成的位图版六轮证据')
        tier=matches[0];links=tier['per_round']
        r.require(coverage.point_decision(links,a.docs)==tier['decision'],'位图版该规模判定不能重算')
    proofs[str(evidence_path.resolve())]=digest(evidence_path)
    r.require(len(links)==6,'位图版基线必须有六轮')
    selected={}
    for row in links:
        for role in ('candidate',)+coverage.cpu_roles(a.docs):
            link=row[role];path=Path(link['directory'])/'report.json'
            r.require(digest(path)==link['report_sha256'],'位图版逐轮报告改变')
            proofs[str(path.resolve())]=digest(path)
            data=json.loads(path.read_text(encoding='utf-8'));command=data['command'];binary=Path(command[0])
            r.require(data['summary']==link['summary'] and digest(binary)==data['runner_sha256'],'位图版逐轮摘要或二进制改变')
            proofs[str(binary.resolve())]=data['runner_sha256']
            if role=='candidate':
                expected=prior['candidate']['command']
                r.require(len(command)==14 and command[5]==str(old['device']) and command[7:12]==['10','8','24','5','30'] and
                    binary.resolve()==Path(expected[0]).resolve() and Path(command[6]).resolve()==Path(expected[6]).resolve(),
                    'NPU基线必须是已验收的位图版运行器和vendor')
            else:
                cpu=role.removesuffix('_single');expected=prior[cpu]['command']
                args=['1','10','1','8','24','5','30'] if role.endswith('_single') else ['32','10','32','8','24','5','30']
                r.require(len(command)==12 and command[5:]==args and binary.resolve()==Path(expected[0]).resolve(),'CPU基线或线程合同改变')
            selected.setdefault('npu' if role=='candidate' else role,data)
    inputs=Path(selected['npu']['command'][1]).parent;description,index=retained.corpus.validate(inputs)
    r.require(description==tier['workload'] and description['docs']==a.docs and description['queries']==8 and
        description['query_length']==3 and description['k']==10,'位图版工作量改变')
    r.require(description['generation']==old['workload']['generation'] and
        {c:description['cases'][c]['terms'] for c in ('A','B')}=={c:old['workload']['cases'][c]['terms'] for c in ('A','B')},'跨规模生成规则或查询ID改变')
    for data in selected.values():
        command=data['command'];r.require(Path(command[1]).resolve()==(inputs/'index').resolve() and
            Path(command[2]).resolve()==(inputs/'A.txt').resolve() and Path(command[3]).resolve()==(inputs/'B.txt').resolve(),'基线CPU/NPU输入路径不同')
    proofs[str(inputs/'preparation_report.json')]=digest(inputs/'preparation_report.json')
    return old,tier,selected,inputs,description,index,proofs


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
        old['affected_checks']['baseline']['output_sha256']==old['affected_checks']['candidate']['output_sha256'],
        '复用候选缺少受影响选择和尾叶检查')
    return old


def build_install(a):
    project=a.output/'project';vendor=a.vendors/source.VENDOR
    r.require(not vendor.exists(),'独立候选 vendor 已存在，停止覆盖')
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
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',args),'实际候选 ABI 改变：'+name)
    for name,raw in source.candidate_sources(a.original).items():
        r.require((project/name).read_bytes()==raw,'候选源码改变：'+name)
    objects=simd.fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    compiler_logs,resources=simd.compiler_resources(a.output)
    return {'vendor':str(vendor),'project':str(project),'metadata':metadata,'device_objects':objects,
        'compiler_logs':compiler_logs,'simt_resources':resources,'missing_resource_fields':'UNKNOWN; no occupancy or spill inference'}


def run(a):
    r.require(sys.platform=='linux' and not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'需要 A5 Linux 和独立新结果目录')
    r.require(all(not a.output.is_relative_to(p) and not p.is_relative_to(a.output) for p in (a.accepted,a.scales,a.original,a.vendors)),'结果目录不能覆盖或嵌套已有证据')
    old,tier,prior,inputs,description,index,proofs=baseline_evidence(a)
    runner=Path(prior['npu']['command'][0]);baseline_vendors=Path(prior['npu']['command'][6]);device=old['device']
    disk_needed=6*(2+len(coverage.cpu_roles(a.docs)))*4*a.docs*8*4+1024**3
    r.require(disk_needed<shutil.disk_usage(a.output.parent).free,'磁盘预算不足')
    plan=retained.layout(a.docs,index['vocab_size'],index['posting_count'])
    r.require(plan['cpu_peak_known_bytes']+512*1024**2<retained.host_available_bytes()*4//5,'主机内存预算不足')
    own=g.checks.sources(OWN+retained.OWN+retained.HOST);a.output.mkdir(parents=True)
    report={'status':'running','scope':'Only positive-bitmap emission changes in accepted large leaf. Reuse score bitmap, threshold, collector, tree, host, CPU and data.',
        'source_sha256':own,'previous_sha256':proofs,'device':device,'docs':a.docs,'workload':description,
        'memory_plan':plan,'disk_budget_bytes':disk_needed,'per_round':[],
        'predeclared_retention_rule':'A/B:6/6 faster than paired accepted BITMAP NPU,>=5% median reduction, lower median p95; pass all declared CPU configurations. Small losses do not stop multi-scale coverage.',
        'baseline_bitmap_report':str(a.accepted/'report.json'),'baseline_scale_report':str((a.accepted if a.docs==10000000 else a.scales)/'report.json')}
    def save():r.write_json(a.output/'report.json',report)
    def check():
        simd.close.scale.unchanged(proofs,'已验收基线或输入改变')
        r.require(g.checks.sources(OWN+retained.OWN+retained.HOST)==own,'候选源码或工具在运行中改变')
        if 'candidate_sha256' in report:simd.close.scale.unchanged(report['candidate_sha256'],'候选安装或工程改变')
    save()
    try:
        atomic=actual_atomic_headers(a.toolkit);report['actual_atomic_header']=atomic;proofs[atomic['path']]=atomic['sha256']
        proofs.update(atomic['files']);save()
        cpuinfo=Path('/proc/cpuinfo').read_bytes();(a.output/'cpu_info.txt').write_bytes(cpuinfo)
        report['host_context']={'cpu_info_sha256':digest(a.output/'cpu_info.txt'),'allowed_cpu_affinity':sorted(os.sched_getaffinity(0)),
            'openmp_environment':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','GOMP_'))}}
        report['management_before']=retained.accepted.prior.management_snapshot(a.output,'sparse_pack_before');save()
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
            report['affected_checks']=reused['affected_checks']
            report['affected_checks_reused_from']=str(a.candidate_from/'report.json')
            save();check()
        else:
            env=g.environment(vendors);selector=ROOT/'results/fused_simd_close2/selector_host_build/bm25_fused_simd_selector_check'
            r.require(str(selector.resolve()) in proofs,'54项原始分数检查器未被既有证据锁定')
            g.execute([selector,vendors,device,a.output/'selector_checks'],a.output/'selector.log',env)
            raw=json.loads((a.output/'selector_checks/report.json').read_text(encoding='utf-8'))
            r.require(raw['status']=='actual_simd_selector_guards_and_strict_bit_rankings_passed' and raw['calls']==54,'原始分数、零值、同分或保护区检查失败')
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
            check()
            result.update(command=command,runner_sha256=digest(Path(command[0])),log_sha256=digest(path/'run.log'))
            r.write_json(path/'report.json',result)
            print(f'N={a.docs} {role} 完整 Search：'+json.dumps(result['summary'],ensure_ascii=False),flush=True)
            return {'directory':str(path),'summary':result['summary'],'report_sha256':digest(path/'report.json')}
        for i in range(6):
            retained.corpus.validate(inputs)
            order=coverage.process_order(a.docs,i)
            row={'round':i+1,'process_order':list(order)}
            for role in order:row[role]=measure(role,a.output/f'round{i+1}'/role)
            for checkpoint in g.checks.CHECKPOINTS:
                left=Path(row['baseline']['directory'])/'actual'/checkpoint;right=Path(row['candidate']['directory'])/'actual'/checkpoint
                for file in left.iterdir():
                    if file.suffix=='.bin':r.require(retained.same_stream(file,right/file.name),'完整分数或树输出位不同')
            report['per_round'].append(row);save()
        retained.corpus.validate(inputs);report['decision']=decision(report['per_round'],a.docs)
        report['management_after']=retained.accepted.prior.management_snapshot(a.output,'sparse_pack_after')
        report['status']=PASS if report['decision']['retain_candidate'] else MISS;check();save()
        print('稀疏 Pack 六轮判定：'+json.dumps(report['decision'],ensure_ascii=False),flush=True)
        print('原基线保持安装；只有全部声明门槛通过才保留候选。',flush=True)
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
    r.require(need<shutil.disk_usage(a.output.parent).free,'全部七档不可变诊断输出的磁盘预算不足')
    a.output.mkdir(parents=True)
    report={'status':'running','baseline':'accepted_score_bitmap','planned_sizes':list(sizes),
        'disk_budget_bytes':need,'tiers':[],'scope':'New sparse emission only. Build once; reuse affected edge checks; fresh paired full Search at every scale. Exit0 means coverage, not every scale winning.'}
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
        report['status']='actual_sparse_pack_all_scale_coverage_checked'
        report['stable_cpu_winning_docs']=[t['docs'] for t in report['tiers'] if t['decision']['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']]
        report['retained_candidate_docs']=[t['docs'] for t in report['tiers'] if t['decision']['retain_candidate']]
        save();print('稀疏 Pack 全规模覆盖：'+json.dumps(report,ensure_ascii=False),flush=True)
        print('Exit0仅表示七档覆盖完成；NPU是否赢CPU、是否优于已验收位图版，分别查看判定。',flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,path in {'accepted':'results/score_bitmap_large_close1','scales':'results/score_bitmap_scales1','original':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'))
    p.add_argument('--docs',type=int,help='只测一个已验收规模；默认完成全部七档')
    p.add_argument('--candidate-from',type=Path,help='复用本脚本已完成的独立候选报告目录，不重编译或重做旧边界检查')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:
        if a.docs is not None:
            r.require(a.docs in coverage.SIZES+(10000000,),'只能测已验收的七个规模');return run(a)
        return sweep(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；保留基线，不生成替代设备结果。',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())

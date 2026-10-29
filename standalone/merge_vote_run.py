"""merge_vote.py 的A5入口；只测262K，Leaf和CPU保持不变。"""
import argparse,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import merge_vote as e

ROOT=e.ROOT;g=e.g;r=e.r;simd=e.simd;retained=e.retained;coverage=e.coverage
source=e.source;builder=e.builder;prior=e.prior
OWN=e.OWN+('standalone/merge_vote_run.py',)


def run(a):
    r.require(sys.platform=='linux' and a.docs==262144,'只在A5验收262K')
    r.require(not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'需要独立结果目录')
    r.require(all(not a.output.is_relative_to(p) and not p.is_relative_to(a.output)
                  for p in (a.completed,a.original,a.leaf_original,a.vendors)),'结果目录不能覆盖或嵌套已有证据/安装')
    r.require(not (a.vendors/source.VENDOR).exists(),'独立Merge候选vendor已存在，保留已有结果，停止覆盖')
    old,runs,inputs,description,index,proofs=e.baseline_evidence(a)
    runner=Path(runs['npu']['command'][0]);base_vendors=Path(runs['npu']['command'][6]);device=old['device']
    needed=6*4*4*a.docs*description['queries']*4+1024**3
    r.require(needed<shutil.disk_usage(a.output.parent).free,'六轮独立诊断输出的磁盘预算不足')
    own=g.checks.sources(OWN+retained.OWN+retained.HOST);a.output.mkdir(parents=True)
    report={'status':'running','source_sha256':own,'previous_sha256':proofs,'device':device,'docs':a.docs,
        'workload':description,'per_round':[],'scope':'Merge-only; Warp Append leaf and CPU unchanged',
        'gate':'6/6 over Warp Append, >=5% median reduction, lower p95, and all declared CPU gates'}
    def save():r.write_json(a.output/'report.json',report)
    def check():
        simd.close.scale.unchanged(proofs,'基线证据改变')
        r.require(g.checks.sources(OWN+retained.OWN+retained.HOST)==own,'运行中源码改变')
        if 'candidate_sha256' in report:simd.close.scale.unchanged(report['candidate_sha256'],'候选安装改变')
    save()
    try:
        report['candidate']=e.build_operator(a,'merge',a.original);check()
        vendors=g.aliases(a.output/'runtime/vendors',a.vendors/prior.source.VENDOR,a.vendors/source.VENDOR)
        project=Path(report['candidate']['project'])
        files=[p for p in (a.vendors/source.VENDOR).rglob('*') if p.is_file() and p.suffix in ('.so','.o','.json','.h')]
        files += [project/name for name in source.candidate_sources(a.original)]
        files += [project/name for name in ('bm25_project.json','CMakePresets.json','op_kernel/CMakeLists.txt','build.sh')]
        report['candidate_sha256']={str(p.resolve()):e.digest(p) for p in files};save();check()
        env=g.environment(vendors)
        g.execute([runner,'--merge-check',vendors,device,a.output/'merge_checks'],a.output/'merge_check.log',env)
        checked=json.loads((a.output/'merge_checks/report.json').read_text(encoding='utf-8'))
        r.require(checked['status']=='actual_global_id_merge_tree_boundaries_guards_and_bits_passed' and checked['calls']==36,'36项Merge检查失败')
        report['affected_checks']={'merge':checked,'leaf':str(a.vendors/prior.source.VENDOR),
            'report_sha256':e.digest(a.output/'merge_checks/report.json'),
            'log_sha256':e.digest(a.output/'merge_check.log')};save()
        def measure(role,path):
            path.mkdir(parents=True);out=path/'actual';npu=role in ('baseline','candidate')
            previous=runs['npu'] if npu else runs[role]
            vendor=(vendors if role=='candidate' else base_vendors) if npu else None
            command=prior.command_for(previous,out,vendor);check()
            g.execute(command,path/'run.log',g.environment(vendor) if npu else os.environ.copy(),timeout=3600)
            result=coverage.verify_single_cpu(out,inputs,description,index,role) if role.endswith('_single') else retained.verify_run(out,inputs,description,index,'npu' if npu else role,device)
            result.update(command=command,runner_sha256=e.digest(Path(command[0])),log_sha256=e.digest(path/'run.log'))
            r.write_json(path/'report.json',result);check()
            print(f'{role}：'+json.dumps(result['summary'],ensure_ascii=False),flush=True)
            return {'directory':str(path),'summary':result['summary'],'report_sha256':e.digest(path/'report.json')}
        for i in range(6):
            retained.corpus.validate(inputs)
            order=coverage.process_order(a.docs,i);row={'round':i+1,'process_order':list(order)}
            for role in order:row[role]=measure(role,a.output/f'round{i+1}'/role)
            for checkpoint in g.checks.CHECKPOINTS:
                left=Path(row['baseline']['directory'])/'actual'/checkpoint;right=Path(row['candidate']['directory'])/'actual'/checkpoint
                for file in left.iterdir():
                    if file.suffix=='.bin':r.require(retained.same_stream(file,right/file.name),'输出位不同')
            report['per_round'].append(row);save()
        retained.corpus.validate(inputs);check()
        report['decision']=e.decision(report['per_round'],a.docs)
        report['status']=e.PASS if report['decision']['retain_candidate'] else e.MISS;save()
        print('六轮判定：'+json.dumps(report['decision'],ensure_ascii=False),flush=True)
        return 0 if report['status']==e.PASS else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save()
        logs=list(a.output.rglob('*.log'))
        if logs:
            last=max(logs,key=lambda p:p.stat().st_mtime)
            print('最近日志：'+str(last)+'\n'+last.read_text(encoding='utf-8',errors='replace')[-24000:],flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,path in {'completed':'results/warp_append_scales1','original':'results/Bm25TopKMerge','leaf_original':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors'}.items():
        p.add_argument('--'+name.replace('_','-'),type=Path,default=ROOT/path)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'));p.add_argument('--docs',type=int,default=262144)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:return run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())

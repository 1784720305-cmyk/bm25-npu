"""262K合并候选：Warp Append Leaf不变，只验收Merge唯一赢家快速路径。"""
import json,sys
from pathlib import Path
from types import FunctionType
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import warp_append as prior,large_search as builder,merge_vote_sources as source

g=prior.g;r=prior.r;simd=prior.simd;retained=prior.retained;coverage=prior.coverage
digest=retained.corpus.digest
OWN=('standalone/merge_vote.py','standalone/merge_vote_sources.py','standalone/large_search.py','standalone/large_search_sources.py')+prior.OWN
PASS='actual_merge_unique_vote_candidate_retained'
MISS='actual_merge_unique_vote_correct_but_not_retained'
# Reuse the accepted builder code with an independent globals dictionary.
# Importing this experiment must not change large_search.source or later audits.
build_operator=FunctionType(builder.build_operator.__code__,
    {**vars(builder),'source':source},'build_merge_vote_operator')


def decision(rows,n):
    result=prior.decision(rows,n);result['baseline']='accepted_warp_append_leaf_and_original_merge';return result


def baseline_evidence(a):
    sweep_path=a.completed/'report.json';sweep=json.loads(sweep_path.read_text(encoding='utf-8'))
    r.require(sweep.get('status')==prior.CHECKED,'需要已完成的Warp Append七档报告')
    matches=[x for x in sweep['tiers'] if x['docs']==a.docs];r.require(len(matches)==1,'该规模没有Warp Append验收证据')
    tier=matches[0];point_path=(a.completed/f'N{a.docs}'/'report.json').resolve()
    r.require(Path(tier['report']).resolve()==point_path and tier['report_sha256']==digest(point_path),'Warp Append分档报告改变')
    point=json.loads(point_path.read_text(encoding='utf-8'))
    r.require(point.get('status') in (prior.PASS,prior.MISS) and point['docs']==a.docs,'Warp Append分档未完成')
    r.require(point['source_sha256']==g.checks.sources(prior.OWN+retained.OWN+retained.HOST),'Warp Append被测源码改变')
    r.require(prior.decision(point['per_round'],a.docs)==point['decision'],'Warp Append六轮判定不能重算')
    r.require(point['status']==(prior.PASS if point['decision']['retain_candidate'] else prior.MISS) and
        tier['status']==point['status'] and tier['decision']==point['decision'],'Warp Append总报告与分档不一致')
    candidate=point['candidate']
    expected={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in prior.source.candidate_sources(a.leaf_original).items()}
    r.require(candidate['metadata']['source_sha256']==expected and candidate['metadata']['vendor_name']==prior.source.VENDOR,'基线Leaf不是Warp Append')
    trusted=Path(candidate['vendor']).resolve(strict=True)
    r.require(trusted==(a.vendors/prior.source.VENDOR).resolve(strict=True),'Warp Append Leaf安装目录改变')
    proofs={str(sweep_path.resolve()):digest(sweep_path),str(point_path.resolve()):digest(point_path)}
    simd.close.scale.unchanged(point['previous_sha256'],'Warp Append前序证据改变');proofs.update(point['previous_sha256'])
    simd.close.scale.unchanged(point['candidate_sha256'],'Warp Append Leaf安装或工程改变');proofs.update(point['candidate_sha256'])
    links=point['per_round'];r.require(len(links)==6,'Warp Append基线必须有六轮')
    selected={};npu_runner=None;baseline_vendors=None
    for row in links:
        for role in ('candidate',)+coverage.cpu_roles(a.docs):
            link=row[role];path=Path(link['directory'])/'report.json';r.require(digest(path)==link['report_sha256'],'Warp Append逐轮报告改变')
            proofs[str(path.resolve())]=digest(path)
            data=json.loads(path.read_text(encoding='utf-8'));command=data['command'];binary=Path(command[0]).resolve()
            r.require(data['summary']==link['summary'] and digest(binary)==data['runner_sha256'],'Warp Append逐轮摘要或二进制改变')
            r.require(str(binary) in proofs and proofs[str(binary)]==data['runner_sha256'],'逐轮运行器未纳入证据链')
            if role=='candidate':
                r.require(len(command)==14 and command[5]==str(point['device']) and command[7:12]==['10','8','24','5','30'],'NPU基线命令改变')
                if npu_runner is None:npu_runner=binary;baseline_vendors=Path(command[6]).resolve()
                r.require(binary==npu_runner and Path(command[6]).resolve()==baseline_vendors,'NPU基线运行器或vendor不一致')
                r.require((baseline_vendors/'bm25_fused_score_topk').resolve(strict=True)==trusted,'基线Leaf别名改变')
                r.require((baseline_vendors/'bm25_topk_merge').resolve(strict=True)==
                    (a.vendors/source.base.MERGE).resolve(strict=True),'基线Merge不是已验收tree安装')
            else:
                args=['1','10','1','8','24','5','30'] if role.endswith('_single') else ['32','10','32','8','24','5','30']
                r.require(len(command)==12 and command[5:]==args,'CPU线程或测量合同改变')
            selected.setdefault('npu' if role=='candidate' else role,data)
    inputs=Path(selected['npu']['command'][1]).parent;description,index=retained.corpus.validate(inputs)
    r.require(description==point['workload'] and description['docs']==a.docs and description['queries']==8 and description['query_length']==3 and description['k']==10,'基线工作量改变')
    for data in selected.values():
        command=data['command']
        r.require(Path(command[1]).resolve()==(inputs/'index').resolve() and Path(command[2]).resolve()==(inputs/'A.txt').resolve() and Path(command[3]).resolve()==(inputs/'B.txt').resolve(),'基线CPU/NPU输入路径不同')
    proofs[str((inputs/'preparation_report.json').resolve())]=digest(inputs/'preparation_report.json')
    return point,selected,inputs,description,index,proofs

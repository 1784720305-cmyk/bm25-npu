"""Build only frozen final operators; bind accepted installed vendors without rebuilding."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import final_support as support,resident_index as r
VENDORS={'fused':'bm25_fused_score_topk_warp_append_large','merge':'bm25_topk_merge_tree'}


def prepare(template,target,kind):
    r.require(not target.exists(),'New project required')
    r.require(not target.resolve().is_relative_to(template.resolve()) and not template.resolve().is_relative_to(target.resolve()),'Template/output overlap')
    r.require((template/'build.sh').is_file(),'Actual CANN-generated build.sh required')
    for directory,dirs,files in os.walk(template,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ('build_out','build','__pycache__','.git')]
        r.require(all(not (Path(directory)/n).is_symlink() for n in dirs+files),'Template symlink rejected')
    shutil.copytree(template,target,ignore=lambda unused,names:[n for n in names if n in ('build_out','build','__pycache__','.git')])
    for path in (ROOT/'operators'/kind).rglob('*'):
        if path.is_file() and path.name!='operator.json':
            dest=target/path.relative_to(ROOT/'operators'/kind);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    preset=target/'CMakePresets.json';data=json.loads(preset.read_text(encoding='utf-8'));count=0
    for p in data.get('configurePresets',[]):
        variables=p.get('cacheVariables',{})
        if 'ASCEND_COMPUTE_UNIT' not in variables:continue
        for key,value in (('ASCEND_COMPUTE_UNIT','ascend950'),('vendor_name',VENDORS[kind])):
            old=variables.get(key)
            if isinstance(old,dict):old['value']=value
            else:variables[key]={'type':'STRING','value':value}
        count+=1
    r.require(count>0,'Actual CANN preset target missing');r.write_json(preset,data)
    r.write_json(target/'final_source.json',{'kind':kind,'vendor':VENDORS[kind],
        'source_sha256':{str(p.relative_to(target)).replace('\\','/'):support.digest(p) for p in target.rglob('*') if p.is_file() and p.suffix in ('.cpp','.h')}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['bind','build'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vendors',type=Path,default=ROOT/'results/custom_opp/vendors')
    p.add_argument('--leaf-template',type=Path,default=ROOT/'results/Bm25FusedScoreTopKUbFix1')
    p.add_argument('--merge-template',type=Path,default=ROOT/'results/Bm25TopKMerge')
    p.add_argument('--install',action='store_true');a=p.parse_args()
    a.output=a.output.resolve();a.vendors=a.vendors.resolve()
    r.require(sys.platform=='linux','Use A5 Linux')
    r.require(not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'New results output required')
    r.require(not a.output.is_relative_to(a.vendors) and not a.vendors.is_relative_to(a.output),'Output/vendors overlap')
    if a.mode=='bind':
        support.aliases(a.output/'vendors',a.vendors/VENDORS['fused'],a.vendors/VENDORS['merge'])
        r.write_json(a.output/'installation.json',{'status':'accepted_vendor_aliases_bound_not_new_device_validation',
            'vendors':{k:str((a.vendors/v).resolve(strict=True)) for k,v in VENDORS.items()}})
        print('Final runtime aliases: '+str(a.output/'vendors'));return
    r.require(not a.install or all(not (a.vendors/v).exists() for v in VENDORS.values()),'Refuse overwriting installed vendors')
    a.output.mkdir(parents=True)
    for kind,template in (('fused',a.leaf_template),('merge',a.merge_template)):
        target=a.output/kind;prepare(template.resolve(),target,kind)
        support.execute(['bash','build.sh'],a.output/(kind+'_build.log'),cwd=target,timeout=1800)
        packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'One actual package required')
        if a.install:support.execute(['bash',packages[0],'--install-path='+str(a.vendors.parent)],a.output/(kind+'_install.log'),timeout=300)
    print('Final packages built; new device validation still required for a new installation.')


if __name__=='__main__':main()

/* Run with runCourseChecks(fixtures), using course-test-fixtures.py output.
   This DOM/URL stand-in checks behavior, not browser layout or URL conformance. */
function runCourseChecks(fixtures) {
  let checks=0;
  const assert=(ok,msg)=>{checks++;if(!ok)throw Error(msg)};
  class Evt {constructor(type){this.type=type}}
  class Node {
    constructor(raw,parent=null){
      this.tagName=raw.tag.toUpperCase();this.attrs={...raw.attrs};this.parentElement=parent;this.handlers={};this.style={};this.hidden='hidden' in this.attrs;this.open='open' in this.attrs;this.checked='checked' in this.attrs;
      this.children=raw.children.map(n=>typeof n==='string'?n:new Node(n,this));this.dataset={};
      Object.entries(this.attrs).forEach(([k,v])=>{if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,a)=>a.toUpperCase())]=v||''});
      this.classList={add:(v)=>this.attrs.class=((this.attrs.class||'')+' '+v),remove:(v)=>this.attrs.class=(this.attrs.class||'').split(' ').filter(x=>x!==v).join(' '),toggle:(v,on)=>on?this.classList.add(v):this.classList.remove(v)};
    }
    get id(){return this.attrs.id||''} get type(){return this.attrs.type||''} get min(){return this.attrs.min} get max(){return this.attrs.max}
    get href(){return this.attrs.href} set href(v){this.attrs.href=v}
    get textContent(){return this.text===undefined?this.children.map(c=>typeof c==='string'?c:c.textContent).join(''):this.text} set textContent(v){this.text=v}
    get options(){return this.querySelectorAll('option')}
    get value(){if(this.val!==undefined)return this.val;if(this.tagName==='SELECT')return (this.options.find(o=>o.hasAttribute('selected'))||this.options[0])?.value||'';return this.attrs.value||''}
    set value(v){this.val=String(v)} get selectedOptions(){return this.options.filter(o=>o.value===this.value)}
    getAttribute(k){return this.attrs[k]??null} hasAttribute(k){return k in this.attrs} setAttribute(k,v){this.attrs[k]=v} removeAttribute(k){delete this.attrs[k]}
    matches(s){
      const attrs=[...s.matchAll(/\[([^=\]]+)(?:="([^"]*)")?\]/g)];
      if(attrs.some(m=>!this.hasAttribute(m[1])||(m[2]!==undefined&&this.attrs[m[1]]!==m[2])))return false;
      s=s.replace(/\[[^\]]*\]/g,'');
      const cls=[...s.matchAll(/\.([\w-]+)/g)];if(cls.some(m=>!(this.attrs.class||'').split(/\s+/).includes(m[1])))return false;
      s=s.replace(/\.[\w-]+/g,'');return !s||s.toUpperCase()===this.tagName;
    }
    querySelectorAll(selector){
      const found=[];const selectors=selector.split(',').map(s=>s.trim().split(/\s+(?![^\[]*\])/));
      const walk=n=>{n.children.forEach(c=>{if(typeof c==='string')return;
        if(selectors.some(parts=>{if(!c.matches(parts.at(-1)))return false;let p=c.parentElement;for(let i=parts.length-2;i>=0;i--){while(p&&!p.matches(parts[i]))p=p.parentElement;if(!p)return false;p=p.parentElement}return true}))found.push(c);walk(c)})};walk(this);return found;
    }
    querySelector(s){return this.querySelectorAll(s)[0]||null} closest(s){let n=this;while(n){if(n.matches(s))return n;n=n.parentElement}return null}
    addEventListener(t,fn){(this.handlers[t]||=[]).push(fn)} dispatchEvent(e){(this.handlers[e.type]||[]).forEach(fn=>fn({type:e.type,target:this}));return true}
    click(){this.dispatchEvent(new Evt('click'))} focus(){} scrollIntoView(){}
  }
  class TestURL {
    constructor(path,base='https://example.test/course/'){
      const b=typeof base==='string'?base:base.href;
      const absolute=/^([a-z]+:)\/\/([^/]+)(.*)$/.exec(path);
      const bm=/^([a-z]+:)\/\/([^/]+)(.*)$/.exec(b);
      this.protocol=absolute?absolute[1]:bm[1];this.origin=this.protocol+'//'+(absolute?absolute[2]:bm[2]);
      let rest=absolute?absolute[3]:path.startsWith('/')?path:(bm[3].split(/[?#]/)[0].replace(/[^/]*$/,'')+path);
      this.hash=rest.includes('#')?'#'+rest.split('#')[1]:'';rest=rest.split('#')[0];const [p,q='']=rest.split('?');const stack=[];
      p.split('/').forEach(x=>{if(x==='..')stack.pop();else if(x!=='.')stack.push(x)});this.pathname=stack.join('/');
      this.params=new Map(q.split('&').filter(Boolean).map(x=>x.split('=').map(decodeURIComponent)));
      this.searchParams={get:k=>this.params.get(k)||null,set:(k,v)=>this.params.set(k,v)};
    }
    get href(){return this.origin+this.pathname+(this.params.size?'?'+[...this.params].map(([k,v])=>encodeURIComponent(k)+'='+encodeURIComponent(v)).join('&'):'')+this.hash}
  }
  function boot(path,{hash='',store=new Map(),blocked=false,via=''}={}){
    const document=new Node(fixtures.pages[path]);document.body=document.querySelector('body');document.documentElement=document.querySelector('html');document.getElementById=id=>document.querySelectorAll('[id]').find(n=>n.id===id)||null;
    const handlers={};const storage={getItem:k=>{if(blocked)throw Error('blocked');return store.get(k)||null},setItem:(k,v)=>{if(blocked)throw Error('blocked');store.set(k,v)},removeItem:k=>store.delete(k)};
    const location={href:'https://example.test/course/'+path+(via?'?via='+via:'')+hash,hash};
    const window={location,localStorage:storage,matchMedia:()=>({matches:false}),addEventListener:(k,f)=>(handlers[k]||=[]).push(f),history:{replaceState:(_,__,v)=>{location.hash=v;location.href=location.href.split('#')[0]+v}},print(){}};
    Object.values(fixtures.scripts).forEach(src=>new Function('window','document','localStorage','Event','URL',src)(window,document,storage,Evt,TestURL));
    document.dispatchEvent(new Evt('DOMContentLoaded'));
    const fire=(event)=>(handlers[event]||[]).forEach(fn=>fn());
    const go=id=>{location.hash='#'+id;location.href=location.href.split('#')[0]+location.hash;fire('hashchange')};
    const set=(id,v,event='change')=>{const n=document.getElementById(id);n.value=v;n.dispatchEvent(new Evt(event))};
    return {document,window,store,go,set,fire,visible:()=>document.querySelectorAll('.step-panel').filter(n=>!n.hidden)};
  }
  const path='lessons/erm/index.html';
  let b=boot(path,{store:new Map([['wildfire-erm-lesson-v1',JSON.stringify({step:'step5',complete:false})]])});
  assert(b.visible()[0].id==='step5','directory resume must keep saved task');
  assert(JSON.parse(b.store.get('wildfire-erm-lesson-v1')).step==='step5','must not overwrite saved task');
  assert(b.visible()[0].querySelector('[data-task-next]').href==='#step6','study must skip practical');
  b.set('erm-mode','practical');
  assert(b.visible()[0].querySelector('[data-task-next]').href==='#step5-assets','practical must start at assets');
  for(const id of ['step5-assets','step5-row-a','step5-smoke','step5-run','step5-verify','step6']){b.go(id);assert(b.visible().length===1&&b.visible()[0].id===id,'practical sequence '+id)}
  const store=b.store;b=boot(path,{store});assert(b.visible()[0].id==='step6','resume practical interpretation');
  b=boot(path,{hash:'#step5-run',store:new Map([['wildfire-interactions-erm-v1',JSON.stringify({controls:{'erm-mode':'study'}})]])});
  assert(b.visible()[0].id==='step5-run'&&b.document.getElementById('erm-mode').value==='practical','explicit practical link overrides saved study');
  b=boot(path,{hash:'#step4'});b.set('repair-budget','other');b.document.querySelector('[data-prediction="no"]').click();
  b=boot(path,{store:b.store});assert(b.visible()[0].id==='step4','saved quiz task');assert(b.document.getElementById('repair-budget').value==='other','saved quiz choice');assert(b.document.getElementById('feedback-budget').textContent.includes('AP is not proportional'),'saved quiz feedback');
  assert(b.document.getElementById('prediction-feedback').textContent.includes('additional training'),'saved prediction feedback');
  b.document.getElementById('reset-progress').click();assert(b.visible()[0].id==='step1','restart task');assert(b.document.getElementById('repair-budget').value==='','restart control');assert(b.document.getElementById('prediction-feedback').textContent==='','restart prediction');
  for(const slug of ['forecast','prediction','evaluation','fold2','missingness','schedule','specialists']){
    b=boot(`lessons/${slug}/index.html`,{hash:'#step3'});assert(b.visible()[0].id==='step3',slug+' explicit step');
    const panels=b.document.querySelectorAll('.step-panel');for(const panel of panels){assert(!!panel.querySelector('.task-purpose'),slug+' task purpose');const next=panel.querySelector('[data-task-next]');assert(next&&!next.textContent.includes('Next step'),slug+' named action')}
    b=boot(`lessons/${slug}/index.html`,{store:b.store});assert(b.visible()[0].id==='step3',slug+' resume');
    b.fire('beforeprint');assert(b.visible().length===panels.length,'print all');b.fire('afterprint');assert(b.visible().length===1&&b.visible()[0].id==='step3','print restores');
  }
  for(const slug of ['forecast','prediction','evaluation','missingness']){
    b=boot(`lessons/${slug}/index.html`);const prefix='real-lab-'+slug;
    assert(!b.document.getElementById(prefix+'-explore').open,'exploration closed initially');
    b.set(prefix+'-event','fire_24935874');b.set(prefix+'-view','fire-mask');b.set(prefix+'-zoom','2.5','input');
    b=boot(`lessons/${slug}/index.html`,{store:b.store});assert(b.document.getElementById(prefix+'-view').value==='fire-mask','restore view');assert(b.document.querySelector('[data-lab-image]').src.includes('fire_24935874/fire-mask.png'),'restore image');assert(b.document.querySelector('[data-lab-image]').style.width==='250%','restore zoom');
    b.document.querySelector('[data-lab-reset]').click();assert(b.document.getElementById(prefix+'-event').value==='fire_24935867','reset event');assert(b.document.querySelector('[data-lab-image]').style.width==='100%','reset zoom');
  }
  b=boot('handbook/student-bridge/index.html',{hash:'#handover',via:'erm.step5-assets'});
  const returns=b.document.querySelectorAll('[data-course-return-link]');assert(returns.length>2,'return in chapter footers');returns.forEach(n=>assert(!n.hidden&&n.href.endsWith('/lessons/erm/index.html#step5-assets'),'exact task return'));
  const nested=b.document.querySelectorAll('a[href]').find(n=>n.getAttribute('href').includes('data-preparation/index.html'));
  assert(nested.href.includes('via=erm.step5-assets'),'nested support keeps task');
  b=boot('handbook/method-recipes/index.html',{hash:'#2-train-one-matched-set',via:'erm.step5-run'});assert(b.visible().length===1&&b.visible()[0].id==='chapter-3','deep reference anchor');assert(!b.visible()[0].querySelector('[data-course-return-link]').hidden,'reference footer return');
  for(const via of ['erm.step999','erm.step5-unknown','evil.step1','erm.step5-run/../../x']){b=boot('handbook/student-bridge/index.html',{via});assert(b.document.querySelector('[data-course-return]').hidden,'invalid return route')}
  b=boot(path,{blocked:true});assert(b.visible()[0].id==='step1','blocked storage works');b.go('step2');assert(b.visible()[0].id==='step2','blocked navigation works');
  b=boot(path,{store:new Map([['wildfire-erm-lesson-v1','broken'],['wildfire-interactions-erm-v1','[]']])});assert(b.visible()[0].id==='step1','corrupt state fallback');
  b=boot('index.html');assert(b.document.querySelectorAll('.hero .button').length===1,'one home action');assert(b.document.querySelector('[data-continue]').href==='lessons/forecast/index.html','new student starts lesson1');
  b=boot('index.html',{store:new Map([['wildfire-course-last-v1',JSON.stringify({page:'erm',step:'step5-run'})]])});assert(b.document.querySelector('[data-continue]').href.endsWith('#step5-run'),'home resumes exact task');
  b=boot(path,{hash:'#repair-budget-reasoning',store:new Map([['wildfire-interactions-erm-v1',JSON.stringify({details:{'repair-budget-reasoning':false}})]])});
  assert(b.visible()[0].id==='step4'&&b.document.getElementById('repair-budget-reasoning').open,'explicit disclosure beats saved closed state');
  b=boot(path,{hash:'#step5'});b.set('erm-mode','practical');b.go('step5-run');b.document.querySelector('[data-study-switch]').click();b.go('step6');
  assert(b.document.getElementById('erm-mode').value==='study','pause practical selects study');assert(b.visible()[0].querySelector('[data-task-prev]').href==='#step5','study returns to plan');
  b=boot('lessons/forecast/index.html',{hash:'#step3'});b.store.set('wildfire-course-last-v1',JSON.stringify({page:'erm',step:'step2'}));b.fire('pageshow');
  assert(JSON.parse(b.store.get('wildfire-course-last-v1')).page==='forecast','history restoration updates last lesson');
  b=boot('index.html');b.store.set('wildfire-course-last-v1',JSON.stringify({page:'erm',step:'step4'}));b.fire('pageshow');assert(b.document.querySelector('[data-continue]').href.endsWith('#step4'),'home history restoration updates continue');
  for(const slug of ['forecast','prediction','evaluation','fold2','missingness','erm','schedule','specialists']){
    b=boot(`lessons/${slug}/index.html`);
    const quizzes=b.document.querySelectorAll('[data-quiz], [data-repair]');
    for(const q of quizzes){
      for(const option of q.options.filter(o=>o.value)){
        b.set(q.id,option.value);const target=q.dataset.repair?'feedback-'+q.dataset.repair:q.id+'-feedback';
        assert(b.document.getElementById(target).textContent.includes(option.dataset.reason),'specific quiz feedback '+q.id);
      }
    }
    const old=b;b=boot(`lessons/${slug}/index.html`,{store:old.store});
    for(const q of quizzes)assert(b.document.getElementById(q.id).value===q.value,'quiz reload '+q.id);
  }
  for(const [slug,id,value,output,expected] of [
    ['prediction','head-weight','0','logit-value','-0.30'],
    ['evaluation','metric-threshold','0.7','metric-counts','τ=0.70'],
    ['schedule','schedule-progress','1500','cosine-value','0.500'],
    ['specialists','route-cost','fixed','route-cost-readout','6,000']]){
    b=boot(`lessons/${slug}/index.html`);b.set(id,value,id==='route-cost'?'change':'input');
    b=boot(`lessons/${slug}/index.html`,{store:b.store});assert(b.document.getElementById(output).textContent.includes(expected),'restored computed output '+id);
  }
  return {checks,passed:true,scope:'Actual generated HTML and scripts; simulated DOM and URL, not browser rendering'};
}

if (typeof module !== 'undefined' && module.exports && require.main === module) {
  const fixtures = JSON.parse(require('fs').readFileSync(0, 'utf8'));
  console.log(JSON.stringify(runCourseChecks(fixtures), null, 2));
}

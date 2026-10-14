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
    click(){this.dispatchEvent(new Evt('click'))} focus(){} scrollIntoView(){this.scrolled=true} getBoundingClientRect(){return {top:this.top??1000}}
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
    const window={location,requestAnimationFrame:fn=>fn(),localStorage:storage,matchMedia:()=>({matches:false}),addEventListener:(k,f)=>(handlers[k]||=[]).push(f),history:{replaceState:(_,__,v)=>{location.hash=v;location.href=location.href.split('#')[0]+v}},print(){}};
    Object.values(fixtures.scripts).forEach(src=>new Function('window','document','localStorage','Event','URL',src)(window,document,storage,Evt,TestURL));
    document.dispatchEvent(new Evt('DOMContentLoaded'));
    const fire=(event)=>(handlers[event]||[]).forEach(fn=>fn());
    const go=id=>{location.hash='#'+id;location.href=location.href.split('#')[0]+location.hash;fire('hashchange')};
    const set=(id,v,event='change')=>{const n=document.getElementById(id);n.value=v;n.dispatchEvent(new Evt(event))};
    return {document,window,store,go,set,fire,visible:()=>document.querySelectorAll('.step-panel').filter(n=>!n.hidden)};
  }
  let b;
  for(const slug of ['forecast','prediction','evaluation','fold2','missingness','erm','schedule','specialists']) {
    const path=`lessons/${slug}/index.html`;
    b=boot(path,{hash:'#step3'});
    const panels=b.document.querySelectorAll('.step-panel');
    assert(b.visible().length===panels.length,slug+' all sections visible at deep link');
    assert(b.window.CourseReading.current()==='step3',slug+' current anchor');
    assert(b.document.querySelectorAll('.step-panel .pager').length===1,'only end-of-lesson navigation');
    assert(b.document.querySelectorAll('.sidebar .small').length===0,'no competing resource menu');
    for(const panel of panels) {
      b.go(panel.id);assert(b.visible().length===panels.length,'anchor does not hide content');
      assert(b.window.CourseReading.current()===panel.id,'anchor updates position');
    }
    b.go('step3');const store=b.store;b=boot(path,{store});
    assert(b.window.CourseReading.current()==='step3','directory resumes saved section');
    assert(b.document.getElementById('step3').scrolled,'resume scrolls to saved section');
    b.document.querySelectorAll('.step-panel').forEach((n,i)=>{n.top=i<4?-100:1000});b.fire('scroll');
    const expected=b.document.querySelectorAll('.step-panel')[3].id;
    assert(b.window.CourseReading.current()===expected,'natural scroll tracks reading');
    assert(JSON.parse(store.get(`wildfire-${slug}-lesson-v1`)).step===expected,'natural scroll saves position');
    b.fire('beforeprint');assert(b.visible().length===panels.length,'print keeps article visible');b.fire('afterprint');assert(b.visible().length===panels.length,'print restore keeps article visible');
    b.document.querySelectorAll('[data-default-open]').forEach(n=>assert(n.open,'required material remains expanded'));
    b.document.getElementById('reset-progress').click();assert(b.window.CourseReading.current()==='step1','restart reading');assert(b.visible().length===panels.length,'restart does not hide sections');
    const quizzes=b.document.querySelectorAll('[data-quiz], [data-repair]');
    for(const q of quizzes)for(const option of q.options.filter(o=>o.value)){
      b.set(q.id,option.value);const target=q.dataset.repair?'feedback-'+q.dataset.repair:q.id+'-feedback';
      assert(b.document.getElementById(target).textContent.includes(option.dataset.reason),'specific feedback');
    }
    b=boot(path,{store:b.store});for(const q of quizzes)assert(b.document.getElementById(q.id).value===q.value,'choices survive reload');
  }
  b=boot('lessons/erm/index.html',{hash:'#step5-run'});assert(b.visible().length===12,'ERM practical and study share one long page');
  assert(!b.document.getElementById('erm-mode'),'no route selector needed to reveal content');
  b=boot('handbook/student-bridge/index.html',{hash:'#handover',via:'erm.step5-assets'});
  assert(b.visible().length===4,'reference is continuous too');
  const returns=b.document.querySelectorAll('[data-course-return-link]');returns.forEach(n=>assert(!n.hidden&&n.href.endsWith('/lessons/erm/index.html#step5-assets'),'exact return'));
  const nested=b.document.querySelectorAll('a[href]').find(n=>n.getAttribute('href').includes('data-preparation/index.html'));assert(nested.href.includes('via=erm.step5-assets'),'nested return context');
  b=boot('handbook/student-bridge/index.html',{store:new Map([['wildfire-course-last-v1',JSON.stringify({page:'forecast',step:'step3'})]])});
  assert(b.document.querySelector('[data-course-return-link]').textContent.startsWith('Continue your last lesson'),'fallback is honestly labelled');
  assert(b.document.querySelector('[data-course-return-link]').href.endsWith('/forecast/index.html#step3'),'direct reference resumes lesson');
  for(const via of ['erm.step999','erm.step5-unknown','evil.step1']){
    b=boot('handbook/student-bridge/index.html',{via});assert(b.document.querySelector('[data-course-return]').hidden,'reject invalid origin');
  }
  for(const slug of ['forecast','prediction','evaluation','missingness']){
    b=boot(`lessons/${slug}/index.html`);const prefix='real-lab-'+slug;
    assert(!b.document.getElementById(prefix+'-explore').open,'optional exploration initially closed');
    b.set(prefix+'-event','fire_24935874');b.set(prefix+'-view','fire-mask');b.set(prefix+'-zoom','2.5','input');
    b=boot(`lessons/${slug}/index.html`,{store:b.store});assert(b.document.querySelector('[data-lab-image]').src.includes('fire_24935874/fire-mask.png'),'restored image');assert(b.document.querySelector('[data-lab-image]').style.width==='250%','restored zoom');
    b.document.querySelector('[data-lab-reset]').click();assert(b.document.querySelector('[data-lab-image]').style.width==='100%','reset image');
  }
  b=boot('lessons/erm/index.html',{blocked:true});assert(b.visible().length===12,'blocked storage permits reading');b.go('step4');assert(b.window.CourseReading.current()==='step4','blocked storage permits anchors');
  b=boot('lessons/erm/index.html',{hash:'#repair-budget-reasoning',store:new Map([['wildfire-interactions-erm-v1',JSON.stringify({details:{'repair-budget-reasoning':false}})]])});assert(b.document.getElementById('repair-budget-reasoning').open,'deep disclosure overrides saved closed state');
  b=boot('index.html');assert(b.document.querySelectorAll('.hero .button').length===1,'single learning entry');
  b=boot('index.html',{store:new Map([['wildfire-course-last-v1',JSON.stringify({page:'erm',step:'step5-run'})]])});assert(b.document.querySelector('[data-continue]').href.endsWith('#step5-run'),'homepage continue');
  for(const path of Object.keys(fixtures.pages)) {
    b=boot(path);const groups=b.document.querySelectorAll('[data-nav-group]');
    assert(groups.length===3,'three navigation groups');
    const active=b.document.querySelectorAll('.sidebar a[aria-current="page"]');
    assert(active.length===(path==='index.html'?0:1),'one actual current page, not a related lesson');
    if(active.length)assert(active[0].closest('[data-nav-group]').open,'current group expanded');
    assert(groups.filter(g=>g.open).length===1,'only relevant group initially expanded');
  }
  b=boot('lessons/forecast/index.html',{hash:'#step3'});
  b.document.dispatchEvent(new Evt('click'));
  const handbookLink=b.document.querySelectorAll('.sidebar a[href]').find(n=>n.href.includes('handbook/student-start/index.html'));
  assert(handbookLink.href.includes('via=forecast.step3'),'sidebar handbook visit retains reading origin');
  const libraryLink=b.document.querySelectorAll('.sidebar a[href]').find(n=>n.href.includes('library/foundations/index.html'));
  assert(libraryLink.href.includes('via=forecast.step3'),'sidebar library visit retains reading origin');
  return {checks,passed:true,scope:'Actual generated HTML and scripts; simulated DOM and URL, not browser rendering'};
}

if (typeof module !== 'undefined' && module.exports && require.main === module) {
  const fixtures = JSON.parse(require('fs').readFileSync(0, 'utf8'));
  console.log(JSON.stringify(runCourseChecks(fixtures), null, 2));
}

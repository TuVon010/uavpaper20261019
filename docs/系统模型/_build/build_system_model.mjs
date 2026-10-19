import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const workspaceDir='E:/postgraduateLife/paper_code/uav_mec_priority';
const base=path.join(workspaceDir,'系统模型');
const build=path.join(base,'_build');
const out=path.join(base,'重绘版');
const SKILL_DIR='C:/Users/13160/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const RUNTIME_PYTHON='C:/Users/13160/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/13160/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
await fs.mkdir(out,{recursive:true});
const W=1440,H=1100;
const p=Presentation.create({slideSize:{width:W,height:H}});
const s=p.slides.add();
s.background.fill='#FFFFFF';
const C={ink:'#16212C',blue:'#1E74B8',purple:'#7850A2',orange:'#AD6216',gray:'#5E6671',light:'#89949F'};
let counter=0;
function shape(g,x,y,w,h,fill='none',stroke='none',sw=0,name=''){
  return s.shapes.add({geometry:g,name:name||`Diagram element ${++counter}`,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:sw,style:'solid'}});
}
function txt(t,x,y,w,h=34,size=24,bold=false,color=C.ink,align='left',fill='none'){
  const a=shape('textbox',x,y,w,h,fill);
  a.text=t;
  a.text.style={fontSize:size,typeface:'Times New Roman',bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:0};
  if(fill!=='none') foregroundText.push(a);
  return a;
}
const foregroundText=[];
const bytes={};
async function asset(name,x,y,w,h,alt,crop){
  const src=name==='satellite.png'?path.join(build,name):path.join(build,'assets',name);
  bytes[src]??=new Uint8Array(await fs.readFile(src));
  const im=s.images.add({blob:bytes[src],contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h},...(crop?{crop}:{} )});
  return im;
}
function anchor(x,y){return shape('rect',x-.5,y-.5,1,1);}
function link(x1,y1,x2,y2,color,width=2.6,style='dashed',arrow=true,name=''){
  const a=anchor(x1,y1),b=anchor(x2,y2);
  const c=s.shapes.connect(a,b,{kind:'straight',line:{fill:color,width,style},...(arrow?{tail:{type:'triangle',width:'med',length:'med'}}:{})});
  if(name)c.name=name;
  c.bringToFront();
  return c;
}
function curve(name,x,y,w,h,commands,color,width=2.4){
  return s.shapes.add({geometry:'custom',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width,style:'solid'},customPaths:[{width:w,height:h,commands}]});
}
function tri(x,y,size,color,rotation){
  const a=shape('triangle',x-size/2,y-size/2,size,size,color);
  a.position={left:x-size/2,top:y-size/2,width:size,height:size,rotation};
  return a;
}
function localLoop(x,y){
  curve('Local computation loop',x,y,62,55,[{moveTo:{x:48,y:9}},{cubicBezTo:{x1:18,y1:-8,x2:-2,y2:10,x:5,y:33}},{cubicBezTo:{x1:14,y1:60,x2:51,y2:55,x:54,y:30}}],C.orange,2.7);
  tri(x+54,y+30,10,C.orange,2);
}
function flight(x,y,flip=false){
  if(!flip){
    curve('UAV horizontal trajectory',x,y,95,55,[{moveTo:{x:2,y:45}},{cubicBezTo:{x1:26,y1:44,x2:24,y2:10,x:53,y:12}},{cubicBezTo:{x1:75,y1:15,x2:76,y2:29,x:91,y:28}}],C.gray,3.2);
    tri(x+91,y+28,12,C.gray,90);
  }else{
    curve('UAV horizontal trajectory',x,y,95,55,[{moveTo:{x:93,y:5}},{cubicBezTo:{x1:66,y1:6,x2:68,y2:42,x:40,y:41}},{cubicBezTo:{x1:19,y1:40,x2:19,y2:27,x:4,y:27}}],C.gray,3.2);
    tri(x+4,y+27,12,C.gray,270);
  }
}
function phone(x,y,index){
  const body=shape('roundRect',x-10,y-17,20,34,'#32495C','#FFFFFF',1.2,`Mobile UD ${index}`);body.borderRadius=3;
  shape('rect',x-6.5,y-12.5,13,22,'#E7F3F8');
  shape('ellipse',x-1.5,y+11.2,3,3,'#FFFFFF');
  txt(`UD ${index}`,x-37,y+21,74,27,21,true,C.ink,'center','#FFFFFF');
  return body;
}

// Reused original scene; transparency is cropped only in the slide placement.
await asset('image34.png',44,369,1352,676,'Post-disaster rescue terrain with damaged buildings and landslide',
  {left:48/1672,top:115/941,right:35/1672,bottom:32/941});
// Relief supplies and rescue workers provide context, not extra computing nodes.
await asset('image12.png',345,657,110,90,'Temporary rescue bridge');
await asset('image9.png',690,776,130,86,'Emergency medical shelter');
await asset('image15.png',811,816,62,54,'Rescue ambulance');
await asset('image35.png',178,700,78,52,'Rescue tent');
await asset('image4.png',588,876,125,83,'Rescue tent');
await asset('image17.png',1090,603,86,53,'Rescue excavator');
await asset('image11.png',270,626,42,63,'Rescue worker');
await asset('image22.png',426,757,43,64,'Mobile user in disaster area');
await asset('image16.png',568,668,42,63,'Mobile user in disaster area');
await asset('image36.png',798,727,47,70,'Mobile user holding a device');
await asset('image10.png',1110,762,47,71,'Medical responder');
await asset('image19.png',1182,611,43,60,'Rescue worker');
await asset('image21.png',486,878,44,66,'Mobile user holding a device');
await asset('image26.png',918,893,44,68,'Mobile user holding a device');

// Three computing UAVs; identical node type is intentionally repeated.
await asset('image27.png',231,288,239,160,'UAV 1 with onboard edge computing',
  {left:30/286,top:19/191,right:27/286,bottom:30/191});
await asset('image37.png',600,351,229,108,'UAV 2 with onboard edge computing',
  {left:36/342,top:16/163,right:32/342,bottom:26/163});
await asset('image29.png',982,287,234,112,'UAV 3 with onboard edge computing');
await asset('image33.png',444,366,40,48,'UAV 1 edge server');
await asset('image33.png',809,409,40,48,'UAV 2 edge server');
await asset('image33.png',1190,350,40,48,'UAV 3 edge server');
txt('UAV 1',260,252,160,34,28,true,C.ink,'center');
txt('UAV 2',632,316,160,34,28,true,C.ink,'center');
txt('UAV 3',1010,252,160,34,28,true,C.ink,'center');
txt('10 MHz, 10 GHz',229,435,263,32,24,false,C.ink,'center','#FFFFFF');
txt('10 MHz, 10 GHz',585,469,277,32,24,false,C.ink,'center','#FFFFFF');
txt('10 MHz, 10 GHz',971,410,273,32,24,false,C.ink,'center','#FFFFFF');
flight(171,366);
flight(532,412);
flight(1181,414,true);

// A single LEO, reachable directly from a ground terminal.
await asset('satellite.png',1020,40,337,145,'Single LEO satellite, AI-generated illustrative asset');
await asset('image33.png',1325,163,39,47,'Satellite edge server');
txt('LEO satellite with edge computing',935,4,470,35,27,true,C.ink,'center');
txt('600 km altitude',1000,188,302,30,24,false,C.ink,'center');
txt('100 MHz shared, 50 GHz',981,220,325,31,24,false,C.ink,'center');

// Only selected, single-hop task paths are illustrated.
link(310,664,327,403,C.blue,3,'dashed',true,'UD 1 offloads directly to UAV 1');
link(459,785,372,408,C.blue,3,'dashed',true,'UD 2 offloads directly to UAV 1');
link(609,682,694,446,C.blue,3,'dashed',true,'UD 3 offloads directly to UAV 2');
link(840,761,755,445,C.blue,3,'dashed',true,'UD 4 offloads directly to UAV 2');
link(1149,783,1106,389,C.blue,3,'dashed',true,'UD 5 offloads directly to UAV 3');
link(1258,647,1380,570,C.purple,3.2,'dashDot',false,'Direct UD 6 to LEO path, segment 1');
link(1380,570,1380,206,C.purple,3.2,'dashDot',false,'Direct UD 6 to LEO path, segment 2');
link(1380,206,1322,164,C.purple,3.2,'dashDot',true,'UD 6 offloads directly to LEO without UAV forwarding');
txt('Direct UD–LEO',1210,290,159,31,22,true,C.purple,'center','#FFFFFF');
txt('offloading',1224,322,133,29,22,false,C.purple,'center','#FFFFFF');

// Representative terminals. All 20 users are described by the shared layer label.
phone(310,664,1);phone(459,785,2);phone(609,682,3);phone(840,761,4);
phone(1149,783,5);phone(1258,647,6);phone(536,910,7);phone(976,924,20);
localLoop(490,875);localLoop(936,887);
txt('Local computing',287,873,197,30,23,false,C.orange,'center','#FFFFFF');
link(477,886,498,897,C.orange,1.5,'solid',false);

// The legend separates communication, mobility, and ideal control signaling.
const legendX=32,legendY=21;
link(legendX,legendY+15,legendX+62,legendY+15,C.blue,3,'dashed');
txt('UD–UAV offloading',111,legendY-2,310,34,25);
link(legendX,legendY+57,legendX+62,legendY+57,C.purple,3,'dashDot');
txt('Direct UD–LEO offloading',111,legendY+40,315,34,25);
link(legendX,legendY+99,legendX+62,legendY+99,C.orange,2.7,'solid');
txt('Local computing',111,legendY+82,300,34,25);
link(legendX,legendY+141,legendX+62,legendY+141,C.gray,3,'solid');
txt('UAV trajectory',111,legendY+124,300,34,25);
link(legendX,legendY+183,legendX+62,legendY+183,C.light,1.8,'dashed',false);
txt('State / control signaling',111,legendY+166,320,34,25);
await asset('image33.png',41,231,35,42,'Legend edge-server symbol');
txt('Edge server',111,232,300,34,25);

// A task attribute annotation, not a queue or a discrete priority class.
txt('Deadline-based task priority',472,24,457,35,27,true);
txt('Task {D, C, τ}',472,68,405,34,27);
txt('Priority ω = 1/τ',472,108,405,35,28,true,C.orange);
txt('Shorter deadline, higher priority',472,150,444,31,24,false);

// This controller is a logical abstraction, not another physical network node.
const ctl=shape('rect',470,208,460,100,'#FFFFFF',C.light,1.6,'Logical centralized decision maker');
ctl.line={fill:C.light,width:1.6,style:'dashed'};
txt('Logical centralized decision maker',483,216,434,30,24,true,C.gray,'center');
txt('Offloading and bandwidth / CPU allocation',483,250,434,26,22,false,C.gray,'center');
txt('UAV trajectory control',483,277,434,26,22,false,C.gray,'center');
link(470,264,466,359,C.light,1.8,'dashed',false,'Ideal controller to aerial layer signaling');
link(930,232,1020,173,C.light,1.8,'dashed',false,'Ideal controller to satellite signaling');

// Compact ground-layer statement outside the terrain.
txt('Post-disaster rescue area: 20 mobile UDs with local computing',170,1019,1100,36,27,true,C.ink,'center','#FFFFFF');
txt('Each task executes at exactly one destination: local, one UAV, or the LEO satellite',100,1057,1240,28,24,false,C.ink,'center');
foregroundText.forEach(a=>a.bringToFront());
s.speakerNotes.text=[
  'System: 20 mobile UDs, three computing UAVs, one directly accessible LEO satellite. Each task is indivisible and executes at a single destination.',
  'Current v3 parameters: each UAV has its own 10 MHz bandwidth pool and 10 GHz CPU. The satellite has a shared 100 MHz pool and 50 GHz CPU. UAV altitude 50 m. LEO altitude 600 km. Per-UAV energy budget 12000 J. Priority is only the reciprocal deadline; no discrete priority classes are modeled.',
  'Representative links show individual task assignments, not simultaneous splitting of a task. No UAV–satellite relay, inter-UAV offloading, inter-satellite links, physical cloud, or extra controller UAV is modeled. The decision maker is a logical ideal-control abstraction.',
  'Model sources: docs/01_系统模型.md, configs/environment.json, manuscript_tvt/sections/system_model.tex. A pre-existing 20 MHz satellite value in project_context.md is stale; current v3 model and configuration use 100 MHz.',
  'Reused visual sources: original user-supplied 系统模型图_论文完善版.pptx. The two supplied published-paper images informed layout only; their networks and graphical assets are not reproduced.',
  'Satellite illustration generated with the built-in ImageGen tool. Original terrain, UAV, people, shelter and server artwork retains its original raster form; labels, task links, phones, curves and logical-controller annotation are native editable PowerPoint objects.'
];
await fs.writeFile(path.join(build,'system_model.layout.json'),await (await s.export({format:'layout'})).text());
await fs.writeFile(path.join(build,'candidate.png'),new Uint8Array(await (await p.export({slide:s,format:'png',scale:1.5})).arrayBuffer()));
const candidatePath=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
console.log('Draft exported');
if(process.argv.includes('--finalize')){
  const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
  const finalPath=path.join(out,'系统模型图_UAV_LEO_优先级重绘版.pptx');
  const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:RUNTIME_PYTHON,
    integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`,'--validate-bullet-geometry','--validate-heading-fit'],
    explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
    fontPolicy:{basis:'design',families:['Times New Roman']},verifyArtifactToolImport:true,
    receiptPath:path.join(build,'final.validation.json')});
  console.log(JSON.stringify(result));
  await fs.writeFile(path.join(out,'系统模型图_UAV_LEO_优先级重绘版.png'),new Uint8Array(await (await p.export({slide:s,format:'png',scale:3})).arrayBuffer()));
}

import {registerHooks,createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import fs from 'node:fs/promises';
const fontRoot='C:/Users/13160/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@pdf-lib/standard-fonts/lib/';
const names={Courier:'Courier',CourierBold:'Courier-Bold',CourierBoldOblique:'Courier-BoldOblique',CourierOblique:'Courier-Oblique',Helvetica:'Helvetica',HelveticaBold:'Helvetica-Bold',HelveticaBoldOblique:'Helvetica-BoldOblique',HelveticaOblique:'Helvetica-Oblique',Symbol:'Symbol',TimesBold:'Times-Bold',TimesBoldItalic:'Times-BoldItalic',TimesItalic:'Times-Italic',TimesRoman:'Times-Roman',ZapfDingbats:'ZapfDingbats',AllEncodings:'all-encodings',Encodings:'all-encodings',encodings:'all-encodings'};
const root='E:/postgraduateLife/paper_code/uav_mec_priority/系统模型/_build/';
const require=createRequire(import.meta.url);
const {Font}=require('@pdf-lib/standard-fonts');
for(const [key,name] of Object.entries(names).filter(([,n])=>n!=='all-encodings')){
  const f=Font.load(name),glyphNames=f.CharMetrics.map(m=>m.N),idx=new Map(glyphNames.map((n,i)=>[n,i]));
  const groups=new Map();
  for(const [a,b,amount] of f.KernPairs){
    if(!idx.has(a)||!idx.has(b))continue;
    if(!groups.has(amount))groups.set(amount,[]);
    groups.get(amount).push(idx.get(a)*glyphNames.length+idx.get(b));
  }
  const kernPairs=[];
  for(const [amount,indices] of groups){let last=0;const deltas=[];for(const i of indices.sort((a,b)=>a-b)){deltas.push(i-last);last=i;}kernPairs.push(amount,deltas);}
  const d={name:f.FontName,bbox:f.FontBBox,ascender:f.Ascender??f.FontBBox[3],descender:f.Descender??f.FontBBox[1],xHeight:f.XHeight??0,capHeight:f.CapHeight??0,glyphNames:glyphNames.join(' '),glyphWidths:f.CharMetrics.map(m=>m.WX),kernPairs};
  await fs.writeFile(root+`pdf_font_${key}.cjs`,'module.exports='+JSON.stringify(d)+';');
}
registerHooks({resolve(specifier,context,nextResolve){
  if(specifier.startsWith('#standard-fonts/')){
    const key=specifier.slice(16);
    if(!names[key])throw new Error('Unknown bundled font alias: '+key);
    return {url:pathToFileURL(root+`pdf_font_${key}.cjs`).href,shortCircuit:true};
  }
  return nextResolve(specifier,context);
}});
const {FileBlob,PresentationFile,registerPdfFontFace}=await import('@oai/artifact-tool');
registerPdfFontFace({family:'Times New Roman',weight:400,style:'normal',source:new Uint8Array(await fs.readFile('C:/Windows/Fonts/times.ttf'))});
registerPdfFontFace({family:'Times New Roman',weight:700,style:'normal',source:new Uint8Array(await fs.readFile('C:/Windows/Fonts/timesbd.ttf'))});
const p=await PresentationFile.importPptx(await FileBlob.load(root+'candidate.pptx'));
const pdf=await p.export({slide:p.slides.items[0],format:'pdf'});
await fs.writeFile(root+'candidate.pdf',new Uint8Array(await pdf.arrayBuffer()));
console.log('PDF exported');

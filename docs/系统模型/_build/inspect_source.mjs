import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';
const root='E:/postgraduateLife/paper_code/uav_mec_priority/系统模型';
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'系统模型图_论文完善版.pptx')));
console.log('slides',p.slides.items.length);
const inspect=await p.inspect({kind:'slide,textbox,shape,image,layout',maxChars:25000});
await fs.writeFile(path.join(root,'_build/source_inspection.ndjson'),inspect.ndjson);
for(let i=0;i<p.slides.items.length;i++){
  const slide=p.slides.items[i];
  const png=await p.export({slide,format:'png',scale:1.3});
  await fs.writeFile(path.join(root,`_build/source_slide_${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
  const lay=await slide.export({format:'layout'});
  await fs.writeFile(path.join(root,`_build/source_slide_${i+1}.json`),await lay.text());
}
console.log('source previews saved');

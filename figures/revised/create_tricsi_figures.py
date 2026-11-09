from pathlib import Path
from html import escape
import subprocess
import zipfile

OUT = Path('/workspace/scratch/5997ce5787a6/TriCSI_Revised_Figures')
OUT.mkdir(exist_ok=True)
NAVY='#183e59'; BLUE='#e9f2f8'; TEAL='#1e655f'; GREEN='#eaf4ef'; INK='#182b38'; GRAY='#536575'

class Figure:
    def __init__(self,w,h,title):
        self.w,self.h=w,h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="#536575"/></marker></defs>',f'<rect width="{w}" height="{h}" fill="white"/>']
        self.text(w/2,43,[title],31,True)
    def text(self,x,y,lines,size=24,bold=False,color=INK,anchor='middle'):
        for i,line in enumerate(lines):
            self.parts.append(f'<text x="{x}" y="{y+i*size*1.36}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(line)}</text>')
    def box(self,x,y,w,h,title,lines=(),fill=BLUE,stroke=NAVY,size=24):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>')
        alln=1+len(lines); total=29+(len(lines))*size*1.36
        yy=y+(h-total)/2+25
        self.text(x+w/2,yy,[title],26,True,stroke)
        if lines:self.text(x+w/2,yy+36,lines,size)
    def arrow(self,points,head=True):
        marker=' marker-end="url(#arrow)"' if head else ''
        self.parts.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="'+GRAY+'" stroke-width="3" stroke-linejoin="round"'+marker+'/>')
    def save(self,stem):
        self.text(self.w/2,self.h-22,['Diagram generated with ChatGPT assistance using Python and SVG, 2026.'],19,color=GRAY)
        svg='\n'.join(self.parts+['</svg>'])
        p=OUT/(stem+'.svg');p.write_text(svg,encoding='utf-8')
        subprocess.run(['inkscape',str(p),'--export-type=png',f'--export-filename={OUT/(stem+".png")}',f'--export-width={self.w*2}'],check=True,capture_output=True)

f=Figure(1500,1000,'Figure 3. CSI preprocessing')
f.box(240,85,1020,110,'INPUT',['LEFT and RIGHT CSI packets, RSSI, and transmitter MAC addresses'])
f.box(70,265,370,125,'Separate wireless links',['Identify LEFT and RIGHT','using transmitter MAC addresses'])
f.box(565,265,370,125,'Prepare measurements',['CSI: amplitude, then log(1 + A)','Retain RSSI for each packet'],size=23)
f.box(1060,265,370,125,'Form time windows',['1 s windows; 0.25 s stride','Resample CSI to 32 × 64'],size=22)
f.arrow([(750,195),(750,225),(255,225),(255,265)])
f.arrow([(440,327),(565,327)]);f.arrow([(935,327),(1060,327)])
f.arrow([(1245,390),(1245,450),(750,450),(750,480)])
f.arrow([(750,450),(270,450),(270,480)])
f.arrow([(1245,450),(1230,450),(1230,480)])
f.box(60,480,420,200,'Absolute CSI map',['Subtract the session EMPTY','median for each subcarrier','One 32 × 64 map per link'])
f.box(540,480,420,200,'Packet-centered CSI map',['Subtract each packet’s median','across its subcarriers','One 32 × 64 map per link'])
f.box(1020,480,420,200,'RSSI summaries',['Mean relative to session EMPTY','Within-window standard deviation','Two values per link'])
f.arrow([(270,680),(270,735),(750,735),(750,795)])
f.arrow([(750,680),(750,795)])
f.arrow([(1230,680),(1230,735),(750,735)],head=False)
f.box(240,795,1020,120,'OUTPUT',['Four 32 × 64 CSI maps and four RSSI summaries','Inputs to the two-link neural model'],fill=GREEN,stroke=TEAL)
f.save('Figure_3_CSI_Preprocessing')

f=Figure(1500,1200,'Figure 4. Two-link TriCSI-Anchor network')
f.text(510,99,['INPUTS'],23,True,GRAY)
xs=[50,380,710]
f.box(xs[0],130,280,105,'LEFT CSI maps',['Two 32 × 64 maps'])
f.box(xs[1],130,280,105,'RIGHT CSI maps',['Two 32 × 64 maps'])
f.box(xs[2],130,280,105,'RSSI summaries',['Four values'])
f.box(xs[0],300,280,110,'LEFT encoder',['64-value representation'])
f.box(xs[1],300,280,110,'RIGHT encoder',['64-value representation'])
f.box(xs[2],300,280,110,'RSSI network',['Learned RSSI features'])
for x in xs:f.arrow([(x+140,235),(x+140,300)])
f.box(245,490,550,105,'Learned link weights and fusion',['Combine LEFT, RIGHT, and RSSI information'],size=23)
for x in xs:f.arrow([(x+140,410),(x+140,450),(520,450),(520,490)])
f.box(245,645,550,100,'Shared representation',['Normalized 64-value vector'])
f.arrow([(520,595),(520,645)])
f.text(520,810,['TRAINING OUTPUTS'],23,True,GRAY)
for x,title,lines in [(50,'Zone head',['10-class prediction']),(380,'Occupancy head',['EMPTY or occupied']),(710,'Coordinate head',['x–y prediction'])]:
    f.box(x,850,280,105,title,lines)
    if x==380:
        f.arrow([(520,745),(520,775),(685,775),(685,902),(660,902)])
    else:
        f.arrow([(520,745),(520,775),(x+140,775),(x+140,850)])
f.text(520,1090,['Zone and coordinate heads provide training supervision.'],22,color=GRAY)
f.box(1060,645,390,140,'Zone-reference similarity',['Compare with nine stored','training-zone prototypes'],fill=GREEN,stroke=TEAL)
f.arrow([(795,695),(1060,695)])
f.text(1255,580,['FROZEN EVALUATION'],23,True,TEAL)
f.box(1060,935,390,165,'FINAL OUTPUT',['Average window probabilities','Use occupancy and zone similarity','Predict EMPTY or Z1–Z9'],fill=GREEN,stroke=TEAL,size=22)
f.arrow([(1255,785),(1255,935)])
f.arrow([(520,955),(520,1015),(1060,1015)])
f.save('Figure_4_TriCSI_Anchor_Network')

readme='''TriCSI revised Figures 3 and 4

PNG files are high-resolution images for insertion into Word.
SVG files are editable vector versions and keep text sharp when resized.
The Python script recreates both figures (requires Inkscape on the system path).

These are programmatically drawn scientific diagrams based on Sections 2.4–2.7
of Science Research Paper 2026-2027(6).docx. No image-generation model was used.

Figure 3 separates absolute CSI correction, packet centering, and RSSI summaries.
Figure 4 distinguishes the three training heads from frozen evaluation using
occupancy and stored training-zone prototypes. Zone and coordinate heads are
training outputs, not the reported final decision rule.

Credit for these newly generated diagrams:
Diagram generated with ChatGPT assistance using Python and SVG, 2026.

Replace the earlier Microsoft PowerPoint credit when using these versions.
Keep the existing scientific captions immediately beneath the figures.
'''
(OUT/'README.txt').write_text(readme)
script=Path(__file__)
archive=OUT.parent/'TriCSI_Revised_Figures_3_and_4.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():z.write(p,p.name)
    z.write(script,script.name)
print(archive)

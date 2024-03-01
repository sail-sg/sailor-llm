from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import XQUADDataset

xquad_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',)


system_prompt = '''โปรดอ่านบริบทที่กำหนดและตอบคำถาม คำตอบของคุณควรเป็นภาษาไทยและง่ายที่สุด\n'''

exp_1 = '''บริบท: สัตว์เลี้ยงลูกด้วยนมที่ใหญ่ที่สุดที่อาศัยอยู่ในระดับความสูงสูงสุดคือแพะภูเขาอัลไพน์ ซึ่งได้รับการมองเห็นได้สูงถึง 3,000 ม. (9,843 ฟุต) แพะภูเขาอาศัยอยู่ในถ้ำและลงมากินหญ้าอัลไพน์อันชุ่มฉ่ำ เลียงผาจัดเป็นละมั่ง มีขนาดเล็กกว่าแพะภูเขาและพบได้ทั่วเทือกเขาแอลป์ อาศัยอยู่เหนือแนวต้นไม้และพบได้ทั่วไปในเทือกเขาอัลไพน์ทั้งหมด พื้นที่ทางตะวันออกของเทือกเขาแอลป์ยังคงเป็นที่อยู่ของหมีสีน้ำตาล ในสวิตเซอร์แลนด์ รัฐเบิร์นตั้งชื่อตามหมี แต่หมีตัวสุดท้ายได้รับการบันทึกว่าถูกฆ่าในปี 1792 เหนือ Kleine Scheidegg โดยนักล่าสามคนจากกรินเดลวาลด์
คำถาม: สัตว์เลี้ยงลูกด้วยนมที่ใหญ่ที่สุดที่อาศัยอยู่ในระดับความสูงสูงสุดคืออะไร?'''
ans_1 = '''คำตอบ: แพะภูเขาอัลไพน์\n'''

exp_2 = '''บริบท: เชลล์ได้รับการบูรณาการในแนวดิ่งและดำเนินงานในทุกด้านของอุตสาหกรรมน้ำมันและก๊าซ รวมถึงการสำรวจและการผลิต การกลั่น การจัดจำหน่ายและการตลาด ปิโตรเคมี การผลิตและการค้าพลังงาน มีกิจกรรมด้านพลังงานหมุนเวียนเล็กน้อยในรูปของเชื้อเพลิงชีวภาพและลม มีการดำเนินงานในกว่า 90 ประเทศ ผลิตน้ำมันได้ประมาณ 3.1 ล้านบาร์เรลต่อวัน และมีสถานีบริการ 44,000 แห่งทั่วโลก บริษัทเชลล์ ออยล์ ซึ่งเป็นบริษัทในเครือในสหรัฐอเมริกา เป็นหนึ่งในธุรกิจที่ใหญ่ที่สุด
คำถาม: เชลล์มีสถานีบริการน้ำมันทั่วโลกกี่แห่ง?'''
ans_2 = '''คำตอบ: 44,000\n'''

exp_3 = '''บริบท: ระบบน้ำร้อนโซล่าร์ใช้แสงแดดในการเทสายะน้ำ ในละติจูดที่ต่ำ (ต่ำกว่า 40 องศา) สามารถให้บริการน้ำร้อนในบ้านได้ถึง 60-70% ด้วยอุณหภูมิสูงสุดที่ 60 °C โดยใช้ระบบน้ำร้อนโซล่าร์ ประเภทที่ใช้มากที่สุดคือ ท่อรวมของท่อ (44%) และท่อรวมแผ่นแบน (34%) ที่ใช้โดยทั่วไปสำหรับน้ำร้อนในบ้าน และท่อรวมพลาสติก (21%) ที่ใช้เป็นส่วนใหญ่ในการทำอุ่นสระว่ายน้ำโดยเฉพาะ
คำถาม: เครื่องทำน้ำอุ่นพลังงานแสงอาทิตย์ชนิดใดที่ใช้ทำความร้อนในสระน้ำ?'''
ans_3 = '''คำตอบ: ท่อรวมพลาสติก\n'''

prompt_input = '''บริบท: {context}
คำถาม: {question}'''
prompt_output = '''คำตอบ:'''

xquad_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt=system_prompt),
            ],
            round=[
                dict(role='HUMAN', prompt=exp_1),
                dict(role='BOT', prompt=ans_1),
                dict(role='HUMAN', prompt=exp_2),
                dict(role='BOT', prompt=ans_2),
                dict(role='HUMAN', prompt=exp_3),
                dict(role='BOT', prompt=ans_3),
                dict(role='HUMAN', prompt=prompt_input),
                dict(role='BOT', prompt=prompt_output),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), batch_size=4, max_out_len=155))

xquad_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role="BOT")

xquad_datasets = [
    dict(
        type=XQUADDataset,
        abbr='xquad-th',
        path='./data/xquad/xquad_th.json',
        reader_cfg=xquad_reader_cfg,
        infer_cfg=xquad_infer_cfg,
        eval_cfg=xquad_eval_cfg)
]
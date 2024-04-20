from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer 
from opencompass.datasets import M3EXAMTHDataset, M3examThEvaluator


m3exam_reader_cfg = dict(
    input_columns=['question_text', 'full_options'],
    output_column='answer')

system_prompt = '''ต่อไปนี้เป็นคำถามแบบปรนัย โปรดระบุเฉพาะตัวเลือกที่ถูกต้องเท่านั้น โดยไม่มีรายละเอียดหรือคำอธิบายอื่นใด\n'''


exp_1 = '''คำถาม: ข้อใดเป็นหน้าที่ของพลเมืองไทยทุกคน
๑. เคารพกฎหมาย
๒. แก้ปัญหาของประเทศ
๓. ลงคะแนนเสียงเลือกตั้ง
๔. เสียภาษีเงินได้บุคคลธรรมดา'''
ans_1 = '''คำตอบ: ๑\n'''


exp_2 = '''คำถาม: บุคคลในข้อใดจัดเป็นอภัพพบุคคล ประเภทบุคคลที่ห้ามอุปสมบทแบบเด็ดขาด
1. คนมีเพศบกพร่อง
2. คนไม่มีอุปัชฌาย์
3. คนยืมบาตรเขามา
4. คนที่ไม่มีบาตรและจีวร
5. คนที่ถือเอาสงฆ์เป็นอุปัชฌาย์'''
ans_2 = '''คำตอบ: 1\n'''


exp_3 = '''คำถาม: ซื้อน้ำผลไม้ 9 กล่อง ราคากล่องละ 60.50 บาท จะต้องจ่ายเงินทั้งหมดกี่บาท
1. 455.40
2. 540.50
3. 544.50
4. 594.45'''
ans_3 = '''คำตอบ: 3\n'''


prompt_input = '''คำถาม: {question_text}
{full_options}'''
prompt_output = '''คำตอบ:'''


m3exam_infer_cfg = dict(
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
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=10))


m3exam_eval_cfg = dict(
    evaluator=dict(type=M3examThEvaluator),
    pred_role="BOT")

m3exam_datasets = [
    dict(
        type=M3EXAMTHDataset,
        abbr='m3exam-th',
        path='./data/m3exam/thai-questions-test.json',
        reader_cfg=m3exam_reader_cfg,
        infer_cfg=m3exam_infer_cfg,
        eval_cfg=m3exam_eval_cfg)
]
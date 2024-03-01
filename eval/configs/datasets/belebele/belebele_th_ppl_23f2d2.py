from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import BelebeleDataset, BelebeleEvaluator


belebele_reader_cfg = dict(
    input_columns=['flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4'],
    output_column='answer')

system_prompt = '''โปรดปฏิบัติตามตัวอย่างที่ให้มา อ่านบริบท และตอบคำถาม\n'''

exp_1 = '''บริบท: สัญลักษณ์นิวเคลียร์ใช้ในการเขียนสมการนิวเคลียร์สำหรับการสลายกัมมันตภาพรังสี ลองพิจารณาตัวอย่างของการสลายตัวแบบเบตาลบของทอเรียม-234 ไปเป็นโปรแทกติเนียม-234 ปฏิกิริยานี้แสดงด้วยสมการ:
คำถาม: อะไรคือสิ่งที่ใช้ในการเขียนสมการนิวเคลียร์สำหรับการสลายกัมมันตภาพรังสี?'''
ans_1 = '''คำตอบ: สัญลักษณ์นิวเคลียร์\n'''


exp_2 = '''บริบท: สินค้าลดราคาใดๆ ที่ซื้อสามารถคืนเป็นเครดิตร้านค้าได้ แต่ไม่สามารถขอคืนเงินตามราคาที่ซื้อได้ จำหน่ายเครื่องใช้ไฟฟ้าภายในบ้านและอุปกรณ์ทำสวนทุกชิ้นพร้อมอุปกรณ์ก่อสร้างที่คัดสรร
คำถาม: หากข้อความข้างต้นเป็นจริง ข้อใดต่อไปนี้จะต้องเป็นจริงด้วย'''
ans_2 = '''คำตอบ: ไม่มีการคืนอุปกรณ์ทำสวนเพื่อขอเงินคืน\n'''


exp_3 = '''บริบท: ฉันมีเจ็ดถุง สามถุงใหญ่และอีกสี่ถุงเล็ก ฉันมีบาสเก็ตบอลและวอลเล่ย์บอล ฉันใส่ลูกวอลเลย์บอลสองลูกไว้ในถุงเล็กแต่ละใบ และฉันใส่ลูกบาสเก็ตบอลสองลูกและลูกวอลเล่ย์บอลสองลูกไว้ในกระเป๋าใบใหญ่แต่ละใบ จำนวนวอลเล่ย์บอลคืออายุของฉัน
คำถาม: บาสเก็ตบอลอยู่ที่ไหน?'''
ans_3 = '''คำตอบ: ในถุงใหญ่.\n'''


prompt_input = '''บริบท: {flores_passage}
คำถาม: {question}'''
prompt_output = '''คำตอบ:'''


belebele_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            chr(ans+64): dict(
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
                    dict(role='BOT', prompt=f'คำตอบ: {{mc_answer{ans}}}'),
                ])
            for ans in range(1, 5)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))


belebele_eval_cfg = dict(
    evaluator=dict(type=BelebeleEvaluator),
    pred_role="BOT")

belebele_datasets = [
    dict(
        type=BelebeleDataset,
        abbr='belebele-th-ppl',
        path='./data/belebele/tha_Thai.jsonl',
        reader_cfg=belebele_reader_cfg,
        infer_cfg=belebele_infer_cfg,
        eval_cfg=belebele_eval_cfg)
]

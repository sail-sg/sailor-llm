from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import BelebeleDataset, BelebeleEvaluator


belebele_reader_cfg = dict(
    input_columns=['flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4'],
    output_column='answer')

system_prompt = '''Hãy làm theo các ví dụ đã cho, đọc ngữ cảnh và trả lời câu hỏi.\n'''

exp_1 = '''Bối cảnh: Ký hiệu hạt nhân được dùng để viết phương trình hạt nhân của sự phân rã phóng xạ. Hãy xem xét ví dụ về sự phân rã beta-trừ của thorium-234 thành protactinium-234. Phản ứng này được biểu diễn bằng phương trình:.
Câu hỏi: Người ta dùng gì để viết phương trình hạt nhân của sự phân rã phóng xạ?'''
ans_1 = '''Trả lời: ký hiệu hạt nhân\n'''


exp_2 = '''Bối cảnh: Bất kỳ mặt hàng giảm giá nào đã mua đều có thể được trả lại để lấy tín dụng của cửa hàng nhưng không được hoàn lại giá mua. Mọi thiết bị gia dụng và mọi thiết bị làm vườn đều được giảm giá cùng với các dụng cụ xây dựng được chọn lọc.
Câu hỏi: Nếu những câu trên là đúng thì câu nào sau đây cũng phải đúng?'''
ans_2 = '''Trả lời: Không có thiết bị làm vườn nào được trả lại để được hoàn lại tiền.\n'''


exp_3 = '''Bối cảnh: Tôi có bảy túi. Ba túi lớn và bốn túi còn lại nhỏ. Tôi có một số quả bóng rổ và bóng chuyền. Tôi để hai quả bóng chuyền vào mỗi túi nhỏ. Và tôi để hai quả bóng rổ và hai quả bóng chuyền vào mỗi chiếc túi lớn. Số quả bóng chuyền bằng tuổi tôi.
Câu hỏi: Bóng rổ ở đâu?'''
ans_3 = '''Trả lời: Trong những chiếc túi lớn.\n'''


prompt_input = '''Bối cảnh: {flores_passage}
Câu hỏi: {question}'''
prompt_output = '''Trả lời:'''


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
                    dict(role='BOT', prompt=f'Trả lời: {{mc_answer{ans}}}'),
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
        abbr='belebele-vi-ppl',
        path='./data/belebele/vie_Latn.jsonl',
        reader_cfg=belebele_reader_cfg,
        infer_cfg=belebele_infer_cfg,
        eval_cfg=belebele_eval_cfg)
]

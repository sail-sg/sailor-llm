from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer 
from opencompass.datasets import M3EXAMVIDataset, M3examEmEvaluator


m3exam_reader_cfg = dict(
    input_columns=['question_text', 'full_options'],
    output_column='answer_text')

system_prompt = '''Sau đây là những câu hỏi trắc nghiệm, vui lòng chỉ đưa ra phương án đúng, không kèm theo bất kỳ chi tiết hay giải thích nào khác.\n'''


exp_1 = '''Câu hỏi: Trong các câu sau đây, câu nào không chứa thành phần tình thái?
A. Nhiều mây đấy.
B. Trời ơi, chỉ còn năm phút.
C. Đêm khuya, chó sủa nhiều chắc là có trộm.
D. Hình như thu đã về.'''
ans_1 = '''Trả lời: B\n'''


exp_2 = '''Câu hỏi: Hỗn số $5\\frac{3}{5}$ được viết dưới dạng phân số là:
A. $21/5$
B. $25/3$
C. $13/10$
D. $28/5$'''
ans_2 = '''Trả lời: D\n'''


exp_3 = '''Câu hỏi: Xét về cấu tạo ngữ pháp, câu “Tác phẩm vừa là kết tinh của tâm hồn người sáng tác, vừa là sợi dây truyền cho mọi người sự sống mà nghệ sĩ mang trong lòng\" (Nguyễn Đình Thi Tiếng nói của văn nghệ) thuộc kiểu câu nào?
A. Câu đơn.
B. Câu ghép.
C. Câu đặc biệt.
D. Câu rút gọn.'''
ans_3 = '''Trả lời: A\n'''


prompt_input = '''Câu hỏi: {question_text}
{full_options}'''
prompt_output = '''Trả lời:'''


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
    evaluator=dict(type=M3examEmEvaluator),
    pred_role="BOT")

m3exam_datasets = [
    dict(
        type=M3EXAMVIDataset,
        abbr='m3exam-vi',
        path='./data/m3exam/vietnamese-questions-test.json',
        reader_cfg=m3exam_reader_cfg,
        infer_cfg=m3exam_infer_cfg,
        eval_cfg=m3exam_eval_cfg)
]
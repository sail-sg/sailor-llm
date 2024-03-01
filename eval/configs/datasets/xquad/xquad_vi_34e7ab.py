from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import XQUADDataset

xquad_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',)


system_prompt = '''Hãy đọc ngữ cảnh cho sẵn và trả lời câu hỏi. Câu trả lời của bạn phải bằng tiếng Việt và càng đơn giản càng tốt.\n'''

exp_1 = '''Bối cảnh: Động vật có vú lớn nhất sống ở độ cao cao nhất là dê núi Alpine, đã được quan sát tại độ cao lên đến 3.000 m (9.843 ft). Dê núi sống trong hang động và đi xuống để ăn cỏ núi Alpine mềm ngon. Được phân loại là loài linh dương, dê núi Alpine nhỏ hơn so với dê chamois và được tìm thấy khắp nơi trên dãy núi Alps, sống trên đường biên cây và phổ biến trong toàn bộ dãy núi Alpine. Các khu vực ở phía đông của dãy Alps vẫn là nơi ở của gấu nâu. Ở Thụy Sĩ, bang Bern được đặt tên theo gấu nhưng con gấu cuối cùng được ghi nhận đã bị giết vào năm 1792 trên Kleine Scheidegg bởi ba thợ săn từ Grindelwald.
Câu hỏi: Động vật có vú lớn nhất sống ở độ cao cao nhất là gì?'''
ans_1 = '''Trả lời: dê núi Alpine\n'''

exp_2 = '''Bối cảnh: Shell được tích hợp theo chiều dọc và hoạt động trong mọi lĩnh vực của ngành công nghiệp dầu và khí đốt, bao gồm khám phá và sản xuất, chế biến, phân phối và tiếp thị, hóa dầu, phát điện và giao dịch. Công ty cũng có hoạt động năng lượng tái tạo nhỏ trong các loại nhiên liệu sinh học và gió. Shell hoạt động trong hơn 90 quốc gia, sản xuất khoảng 3.1 triệu thùng dầu tương đương mỗi ngày và có 44,000 trạm dịch vụ trên toàn thế giới. Công ty Dầu Shell, công ty con của nó tại Hoa Kỳ, là một trong những doanh nghiệp lớn nhất của Shell.
Câu hỏi: Shell có bao nhiêu trạm dịch vụ trên toàn thế giới?'''
ans_2 = '''Trả lời: 44,000\n'''

exp_3 = '''Bối cảnh: Hệ thống nước nóng năng lượng mặt trời sử dụng ánh sáng mặt trời để làm nóng nước. Ở các vĩ độ địa lý thấp (dưới 40 độ), hệ thống sưởi ấm bằng năng lượng mặt trời có thể cung cấp từ 60 đến 70% lượng nước nóng sinh hoạt có nhiệt độ lên tới 60 °C. Các loại máy nước nóng năng lượng mặt trời phổ biến nhất là bộ thu ống chân không (44%) và bộ thu tấm phẳng tráng men (34%) thường được sử dụng cho nước nóng sinh hoạt; và những người thu gom nhựa không tráng men (21%) được sử dụng chủ yếu để sưởi ấm bể bơi.
Câu hỏi: Loại máy nước nóng năng lượng mặt trời nào được sử dụng để làm nóng bể bơi?'''
ans_3 = '''Trả lời: người thu gom nhựa không tráng men\n'''

prompt_input = '''Bối cảnh: {context}
Câu hỏi: {question}'''
prompt_output = '''Trả lời:'''

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
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), max_out_len=95))

xquad_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role="BOT")

xquad_datasets = [
    dict(
        type=XQUADDataset,
        abbr='xquad-vi',
        path='./data/xquad/xquad_vi.json',
        reader_cfg=xquad_reader_cfg,
        infer_cfg=xquad_infer_cfg,
        eval_cfg=xquad_eval_cfg)
]
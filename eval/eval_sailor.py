from mmengine.config import read_base

with read_base():
    from .datasets.xquad.xquad_th_34e7ab import xquad_datasets as xquad_th
    from .datasets.tydiqa_id.tydiqa_id_346aa7 import tydiqa_datasets as tydiqa_id
    from .datasets.xquad.xquad_vi_34e7ab import xquad_datasets as xquad_vi

    from .datasets.xcopa.xcopa_th_ppl_49je23 import xcopa_datasets as xcopa_th_ppl
    from .datasets.xcopa.xcopa_id_ppl_49je23 import xcopa_datasets as xcopa_id_ppl
    from .datasets.xcopa.xcopa_vi_ppl_49je23 import xcopa_datasets as xcopa_vi_ppl

    from .datasets.m3exam.m3exam_jv_4fs13f import m3exam_datasets as m3exam_jv
    from .datasets.m3exam.m3exam_th_481ea1 import m3exam_datasets as m3exam_th
    from .datasets.m3exam.m3exam_vi_172ds2 import m3exam_datasets as m3exam_vi

    from .datasets.m3exam.m3exam_th_ppl2_481ea1 import m3exam_datasets as m3exam_th_ppl2
    from .datasets.m3exam.m3exam_th_ppl4_481ea1 import m3exam_datasets as m3exam_th_ppl4
    from .datasets.m3exam.m3exam_th_ppl5_481ea1 import m3exam_datasets as m3exam_th_ppl5
    from .datasets.m3exam.m3exam_jv_ppl_4fs13f import m3exam_datasets as m3exam_id_ppl
    from .datasets.m3exam.m3exam_vi_ppl2_172ds2 import m3exam_datasets as m3exam_vi_ppl2
    from .datasets.m3exam.m3exam_vi_ppl3_172ds2 import m3exam_datasets as m3exam_vi_ppl3
    from .datasets.m3exam.m3exam_vi_ppl4_172ds2 import m3exam_datasets as m3exam_vi_ppl4

    from .datasets.belebele.belebele_th_ppl_23f2d2 import belebele_datasets as belebele_th_ppl
    from .datasets.belebele.belebele_id_ppl_23f2d2 import belebele_datasets as belebele_id_ppl
    from .datasets.belebele.belebele_vi_ppl_23f2d2 import belebele_datasets as belebele_vi_ppl


    from .models.sailor_baselines.llama2_7b import models as llama2_7b
    from .models.sailor_baselines.mistral_7b_v0_1 import models as mistral_7b
    from .models.sailor_baselines.seallm_7b_hybrid import models as seallm_7b_hybrid
    from .models.sailor_baselines.seallm_7b_v2 import models as seallm_7b_v2
    from .models.sailor_baselines.qwen1_5_0_5b import models as qwen1_5_0_5b
    from .models.sailor_baselines.qwen1_5_1_8b import models as qwen1_5_1_8b 
    from .models.sailor_baselines.qwen1_5_4b import models as qwen1_5_4b    
    from .models.sailor_baselines.qwen1_5_7b import models as qwen1_5_7b
    from .models.sailor_baselines.qwen1_5_7b_chat import models as qwen1_5_7b_chat

    from .models.sailor.sailor_0_5b import models as sailor_0_5b
    from .models.sailor.sailor_1_8b import models as sailor_1_8b
    from .models.sailor.sailor_4b import models as sailor_4b
    from .models.sailor.sailor_7b import models as sailor_7b


datasets = [*xquad_th, *tydiqa_id, *xquad_vi]
datasets += [*xcopa_th_ppl, *xcopa_id_ppl, *xcopa_vi_ppl]
datasets += [*m3exam_jv, *m3exam_th, *m3exam_vi]
datasets += [*m3exam_th_ppl2, *m3exam_th_ppl4, *m3exam_th_ppl5]
datasets += [*m3exam_id_ppl]
datasets += [*m3exam_vi_ppl2, *m3exam_vi_ppl3, *m3exam_vi_ppl4]
datasets += [*belebele_th_ppl, *belebele_id_ppl, *belebele_vi_ppl]

models = sailor_0_5b + sailor_1_8b + sailor_4b + sailor_7b

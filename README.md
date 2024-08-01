# Visual Chain-of-Thought Multimodal Algorithmic Reasoning
Owned by: Hanyang University MLLAB.

## 🌺 Setting environment (For MLLAB Students)
패키지 및 환경이 갖춰진 도커 이미지를 컨테이너로 굽고, 저장소에 소스 파일을 가져오는 방식.
```bash
docker run -it --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host --name {container_name} -v /media/data2/SMART101/:/data -v {your_home_directory_path}:/SMART101 42a0e9b621e2
git clone git@github.com:J-PARK11/Visual_Chain-of-Thought_Multimodal_Algorithmic_Reasoning.git
pip install -r requirements.txt
```

## 🌄 Setting environment (For KT)

```bash
git clone git@github.com:J-PARK11/Visual_Chain-of-Thought_Multimodal_Algorithmic_Reasoning.git
pip install -r requirements.txt
```

## 🎮 Quick Start
- 스크립트는 크게 *Train*, *Evaluation*, *Data Generation* 세션으로 구성되어 있음.
- 주로 generation.sh로 답변을 생성하며, 이 파일만 Single GPU기반이고, 나머지는 다 DDP 세팅이다.
- 각 스크립트의 주된 내용은 "hf_config.py"에 있는 DataArguments.task에 따라 결정되는데 크게 {'custom', 'supervised', 'zero_shot', 'GT_with_rationale', 'GPT_augmentation_generation', 'GPT_augmentation_train'}으로 구성된다.
- 각 puzzle_list는 학습 혹은 검증에 쓰일 퍼즐의 pid를 의미하고, tot는 instance puzzle의 수를 의미한다.
- 현재 스크립트에 기재된 argument들은 모두 Default로 설정함. 실제 개발할 때 건드리는 파일은 train.sh이다.
- val,test_puzzle_list에 기본 인자로 들어간 퍼즐들은 카테고리 별로 균등하게 넣은 것이다.        
- 모든 task는 python debugging console을 통한 debugging process를 지원한다.

```bash
* Train:
    - ./scripts/train/train.sh                       # 디버깅용 스크립트
    - ./scripts/train/option_train.sh                # 정답 옵션으로 학습.
    - ./scripts/train/gt_rationale_train.sh          # 101개 Root Puzzle 해설로 학습.
    - ./scripts/train/augmented_rationale_train.sh   # GPT로 증강된 해설로 학습.

* Evaluation:
    - ./scripts/eval/eval.sh                         # 테스트 셋 평가 (미완성)
    - ./scripts/eval/generation.sh                   # 답변 생성

* Data Generation:
    - ./scripts/data_generation/GPT_augmentation.sh  # GPT 해설 데이터셋 증강
```

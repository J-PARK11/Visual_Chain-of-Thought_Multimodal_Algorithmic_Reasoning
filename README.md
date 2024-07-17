# Visual Chain-of-Thought Multimodal Algorithmic Reasoning
Owned by: Hanyang University MLLAB.

## 🌺 Setting environment (For MLLAB Students)
패키지 및 환경이 갖춰진 도커 이미지를 컨테이너로 굽고, 저장소에 소스 파일을 가져오는 방식.
```bash
docker run -it --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host --name {container_name} -v /media/data2/SMART101/:/data -v {your_home_directory_path}:/SMART101 42a0e9b621e2
git clone git@github.com:J-PARK11/Visual_Chain-of-Thought_Multimodal_Algorithmic_Reasoning.git
```

## 🎮 Quick Start
train.sh, eval.sh는 DDP, inference.sh는 Single GPU 기반임.
```bash
./scripts/train.sh
./scripts/eval.sh
./scripts/inference.sh
```
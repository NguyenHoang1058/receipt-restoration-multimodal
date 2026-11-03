import torch
import paddle

print("=== KIỂM TRA PYTORCH (Cho VLM, UVDoc, NAFNet) ===")
pytorch_gpu = torch.cuda.is_available()
print(f"Hỗ trợ CUDA: {pytorch_gpu}")
if pytorch_gpu:
    print(f"Số lượng GPU: {torch.cuda.device_count()}")
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

print("\n=== KIỂM TRA PADDLEPADDLE (Cho PaddleOCR) ===")
paddle_gpu = paddle.is_compiled_with_cuda()
print(f"Hỗ trợ CUDA: {paddle_gpu}")
if paddle_gpu:
    print(f"Thiết bị đang dùng: {paddle.get_device()}")
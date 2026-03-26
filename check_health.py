import torch
import torch.nn as nn
from src.models.hybrid_transformer import HidingNetwork, RevealNetwork
from src.models.noise_layer import DifferentiableNoiseLayer
from src.engine.trainer import FFTLoss
from src.core.device import DeviceManager

def run_health_check():
    print("🚀 Starting Nexus-Steg Architecture Health Check...\n")
    
    # 1. Device Check
    device_mgr = DeviceManager()
    device = device_mgr.device
    print(f"[1/5] Device Check: Using {device}")

    # 2. Model Initialization
    try:
        hiding_net = HidingNetwork().to(device)
        reveal_net = RevealNetwork().to(device)
        noise_layer = DifferentiableNoiseLayer().to(device)
        fft_loss_fn = FFTLoss().to(device)
        print("[2/5] Module Initialization: PASSED")
    except Exception as e:
        print(f"[2/5] Module Initialization: FAILED -> {e}")
        return

    # 3. Component Verification (CBAM & Residuals)
    has_cbam = any("cbam" in name.lower() for name, _ in hiding_net.named_modules())
    print(f"[3/5] Component Check: CBAM Detected: {has_cbam}")

    # 4. Forward Pass & Residual Verification
    try:
        # Create dummy 256x256 images
        dummy_cover = torch.randn(1, 3, 256, 256).to(device)
        dummy_secret = torch.randn(1, 3, 256, 256).to(device)
        
        # Test Hiding Network
        stego = hiding_net(dummy_cover, dummy_secret)
        
        # Verify Residual Learning: stego should not be exactly equal to cover, 
        # but should be very close (Cover + 0.1 * Residual)
        diff = torch.abs(stego - dummy_cover).mean().item()
        is_residual = 0 < diff < 0.2
        print(f"[4/5] Forward Pass: PASSED (Residual Scale Diff: {diff:.4f})")
        if not is_residual:
            print("     ⚠️ Warning: Residual learning check suspicious. Ensure stego = cover + 0.1 * residual.")
            
    except Exception as e:
        print(f"[4/5] Forward Pass: FAILED -> {e}")
        return

    # 5. Noise & Robustness Check (RandomResizing)
    try:
        noise_layer.train() # Enable noise
        # Test extreme distortions
        stego_noised = noise_layer(stego)
        revealed = reveal_net(stego_noised)
        
        # Test FFT Loss
        spectral_loss = fft_loss_fn(stego, dummy_cover)
        
        print(f"[5/5] Robustness & FFT Loss: PASSED (FFT Loss: {spectral_loss.item():.4f})")
        print(f"     Output Shape: {revealed.shape}")
    except Exception as e:
        print(f"[5/5] Robustness & FFT Loss: FAILED -> {e}")
        return

    print("\n✅ All systems go! Your unified architecture is ready for training.")

if __name__ == "__main__":
    run_health_check()
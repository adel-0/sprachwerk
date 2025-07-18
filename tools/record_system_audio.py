import pyaudiowpatch as pyaudio
import numpy as np
import sys
import argparse
import wave
import os
import threading
import queue
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import get_outputs_dir


def list_devices():
    """Lists all available audio devices with loopback and microphone detection."""
    print("\n--- Available Audio Devices ---")
    try:
        p = pyaudio.PyAudio()
        loopback_devices = []
        microphone_devices = []
        
        print("Input Devices:")
        for i in range(p.get_device_count()):
            device = p.get_device_info_by_index(i)
            if device['maxInputChannels'] > 0:
                host_api = p.get_host_api_info_by_index(device['hostApi'])
                print(f"  [{i}] {device['name']}")
                print(f"      Channels: {device['maxInputChannels']}, Rate: {device['defaultSampleRate']}, API: {host_api['name']}")
                
                if '[Loopback]' in device['name'] or 'loopback' in device['name'].lower():
                    loopback_devices.append(device)
                    print("      *** LOOPBACK DEVICE (System Audio) ***")
                elif 'microphone' in device['name'].lower() or 'mic' in device['name'].lower():
                    microphone_devices.append(device)
                    print("      *** MICROPHONE DEVICE ***")
                print()
        
        if loopback_devices:
            print("*** LOOPBACK DEVICES FOUND (for system audio) ***")
            for device in loopback_devices:
                print(f"  [{device['index']}] {device['name']}")
        
        if microphone_devices:
            print("*** MICROPHONE DEVICES FOUND ***")
            for device in microphone_devices:
                print(f"  [{device['index']}] {device['name']}")
        
        p.terminate()
    except Exception as e:
        print(f"Error querying devices: {e}")


def get_recording_device(device_index=None):
    """Gets a suitable device for recording system audio."""
    try:
        p = pyaudio.PyAudio()
        
        if device_index is not None:
            if 0 <= device_index < p.get_device_count():
                device = p.get_device_info_by_index(device_index)
                if device['maxInputChannels'] > 0:
                    host_api = p.get_host_api_info_by_index(device['hostApi'])
                    print(f"Using device [{device_index}]: {device['name']} ({host_api['name']})")
                    p.terminate()
                    return device
                else:
                    print(f"Device {device_index} has no input channels.")
            else:
                print(f"Device index {device_index} is out of range.")
            p.terminate()
            return None
        
        # Auto-detect loopback device
        for i in range(p.get_device_count()):
            device = p.get_device_info_by_index(i)
            if device['maxInputChannels'] > 0 and '[Loopback]' in device['name']:
                host_api = p.get_host_api_info_by_index(device['hostApi'])
                if host_api['name'] == 'Windows WASAPI':
                    print(f"Using WASAPI loopback: [{i}] {device['name']}")
                    p.terminate()
                    return device
        
        print("No loopback device found.")
        p.terminate()
        return None
        
    except Exception as e:
        print(f"Error getting device: {e}")
        return None


def get_microphone_device(device_index=None):
    """Gets a suitable microphone device."""
    try:
        p = pyaudio.PyAudio()
        
        if device_index is not None:
            if 0 <= device_index < p.get_device_count():
                device = p.get_device_info_by_index(device_index)
                if device['maxInputChannels'] > 0:
                    host_api = p.get_host_api_info_by_index(device['hostApi'])
                    print(f"Using microphone [{device_index}]: {device['name']} ({host_api['name']})")
                    p.terminate()
                    return device
            p.terminate()
            return None
        
        # Auto-detect microphone device
        for i in range(p.get_device_count()):
            device = p.get_device_info_by_index(i)
            if (device['maxInputChannels'] > 0 and 
                '[Loopback]' not in device['name'] and 
                'stereo mix' not in device['name'].lower() and
                ('microphone' in device['name'].lower() or 'mic' in device['name'].lower())):
                
                host_api = p.get_host_api_info_by_index(device['hostApi'])
                if host_api['name'] == 'Windows WASAPI':
                    print(f"Using WASAPI microphone: [{i}] {device['name']}")
                    p.terminate()
                    return device
        
        print("No microphone device found.")
        p.terminate()
        return None
        
    except Exception as e:
        print(f"Error getting microphone device: {e}")
        return None


def record_audio_stream(p, device, duration, audio_queue, stream_name):
    """Records audio from a single device."""
    stream = None
    try:
        # Find supported sample rate
        supported_rates = [48000, 44100, 32000, 22050, 16000]
        sample_rate = None
        channels = min(device['maxInputChannels'], 2)
        
        for rate in supported_rates:
            try:
                if p.is_format_supported(rate, 
                                          input_device=device['index'], 
                                          input_channels=channels, 
                                          input_format=pyaudio.paInt16):
                    sample_rate = rate
                    break
            except ValueError:
                continue

        if sample_rate is None:
            raise RuntimeError(f"No supported sample rate found for {device['name']}")

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=2048,
            input_device_index=device['index']
        )
        
        frames = []
        total_chunks = int(duration * sample_rate / 2048)
        
        for _ in range(total_chunks):
            data = stream.read(2048, exception_on_overflow=False)
            frames.append(data)

        if frames:
            audio_data = b''.join(frames)
            recording = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to mono if stereo
            if channels > 1:
                recording = recording.reshape(-1, channels).mean(axis=1).astype(np.int16)
            
            audio_queue.put((stream_name, recording, sample_rate))
        else:
            audio_queue.put((stream_name, None, sample_rate))
            
    except Exception as e:
        print(f"Error recording {stream_name}: {e}")
        audio_queue.put((stream_name, None, 44100))
    finally:
        if stream:
            if stream.is_active():
                stream.stop_stream()
            stream.close()


def resample_audio(audio, original_rate, target_rate):
    """Simple resampling using linear interpolation."""
    if original_rate == target_rate:
        return audio
    
    ratio = target_rate / original_rate
    new_length = int(len(audio) * ratio)
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(new_indices, old_indices, audio.astype(np.float32))
    return resampled.astype(np.int16)


def normalize_audio_levels(system_audio, mic_audio, target_rms=0.1):
    """Normalize audio levels to have similar RMS values."""
    if system_audio is None or mic_audio is None:
        return system_audio, mic_audio
    
    # Calculate RMS for each stream
    system_rms = np.sqrt(np.mean((system_audio.astype(np.float32) / 32768.0) ** 2))
    mic_rms = np.sqrt(np.mean((mic_audio.astype(np.float32) / 32768.0) ** 2))
    
    # Avoid division by zero
    if system_rms < 1e-6:
        system_rms = 1e-6
    if mic_rms < 1e-6:
        mic_rms = 1e-6
    
    # Calculate normalization factors
    system_factor = target_rms / system_rms
    mic_factor = target_rms / mic_rms
    
    # Apply normalization with safety limits
    system_factor = min(system_factor, 10.0)  # Max 10x amplification
    mic_factor = min(mic_factor, 10.0)
    
    # Apply normalization
    system_normalized = (system_audio.astype(np.float32) * system_factor).astype(np.int16)
    mic_normalized = (mic_audio.astype(np.float32) * mic_factor).astype(np.int16)
    
    print(f"Level normalization: System x{system_factor:.2f}, Mic x{mic_factor:.2f}")
    
    return system_normalized, mic_normalized


def mix_audio_streams(system_audio, mic_audio, system_rate, mic_rate, system_gain=0.7, mic_gain=1.0, auto_normalize=True, target_level=0.1):
    """Mix system audio and microphone audio with adjustable gains."""
    if system_audio is None and mic_audio is None:
        return None, 44100
    
    if system_audio is None:
        return (mic_audio * mic_gain).astype(np.int16) if mic_audio is not None else None, mic_rate
    
    if mic_audio is None:
        return (system_audio * system_gain).astype(np.int16), system_rate
    
    # Use the higher sample rate
    target_rate = max(system_rate, mic_rate)
    
    # Resample if necessary
    if system_rate != target_rate:
        system_audio = resample_audio(system_audio, system_rate, target_rate)
    
    if mic_rate != target_rate:
        mic_audio = resample_audio(mic_audio, mic_rate, target_rate)
    
    # Ensure same length
    min_length = min(len(system_audio), len(mic_audio))
    system_audio = system_audio[:min_length]
    mic_audio = mic_audio[:min_length]
    
    # Auto-normalize levels if enabled
    if auto_normalize:
        system_audio, mic_audio = normalize_audio_levels(system_audio, mic_audio, target_rms=target_level)
    
    # Mix with gains
    system_float = system_audio.astype(np.float32) * system_gain
    mic_float = mic_audio.astype(np.float32) * mic_gain
    mixed = system_float + mic_float
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 32767:
        mixed = mixed * (32767 / max_val)
    
    return mixed.astype(np.int16), target_rate


def save_as_wav(audio_data, sample_rate, channels, filename=None):
    """Save audio data as WAV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mixed_audio_recording_{timestamp}.wav"
    output_dir = get_outputs_dir()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    try:
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        
        file_size = os.path.getsize(filepath)
        print(f"Saved: {filepath} ({file_size/1024/1024:.2f} MB)")
        return filepath
    except Exception as e:
        print(f"Error saving WAV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Record system audio and microphone simultaneously.")
    parser.add_argument('--system-device', type=int, help='System audio device index (loopback).')
    parser.add_argument('--mic-device', type=int, help='Microphone device index.')
    parser.add_argument('--list-devices', action='store_true', help='List all devices and exit.')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration (default: 5s).')
    parser.add_argument('--output', type=str, help='Output WAV filename.')
    parser.add_argument('--no-save', action='store_true', help='Skip saving WAV file.')
    parser.add_argument('--system-gain', type=float, default=0.7, help='System audio gain (default: 0.7).')
    parser.add_argument('--mic-gain', type=float, default=1.0, help='Microphone gain (default: 1.0).')
    parser.add_argument('--no-normalize', action='store_true', help='Disable automatic level normalization.')
    parser.add_argument('--target-level', type=float, default=0.1, help='Target RMS level for normalization (default: 0.1).')
    parser.add_argument('--system-only', action='store_true', help='Record system audio only.')
    parser.add_argument('--mic-only', action='store_true', help='Record microphone only.')
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    # Get devices
    system_device = None if args.mic_only else get_recording_device(args.system_device)
    mic_device = None if args.system_only else get_microphone_device(args.mic_device)
    
    if not system_device and not mic_device:
        print("\nNo suitable devices found. Try:")
        print("1. List devices: --list-devices")
        print("2. Enable Stereo Mix in sound settings")
        print("3. Use specific devices: --system-device INDEX --mic-device INDEX")
        list_devices()
        sys.exit(1)

    p = None
    try:
        p = pyaudio.PyAudio()
        audio_queue = queue.Queue()
        threads = []
        
        # Start recording threads
        if system_device:
            threads.append(threading.Thread(
                target=record_audio_stream,
                args=(p, system_device, args.duration, audio_queue, "system audio")
            ))
        
        if mic_device:
            threads.append(threading.Thread(
                target=record_audio_stream,
                args=(p, mic_device, args.duration, audio_queue, "microphone")
            ))
        
        for thread in threads:
            thread.start()
        
        print(f"Recording for {args.duration}s...")
        
        for thread in threads:
            thread.join()
        
        # Collect results
        system_audio, mic_audio = None, None
        system_rate, mic_rate = 44100, 44100
        
        while not audio_queue.empty():
            stream_name, audio_data, rate = audio_queue.get()
            if stream_name == "system audio":
                system_audio, system_rate = audio_data, rate
            elif stream_name == "microphone":
                mic_audio, mic_rate = audio_data, rate
        
        # Mix audio
        mixed_audio, final_sample_rate = mix_audio_streams(
            system_audio, mic_audio, system_rate, mic_rate, 
            args.system_gain, args.mic_gain, not args.no_normalize, args.target_level
        )
        
        if mixed_audio is None:
            print("No audio recorded from any source.")
            sys.exit(1)
        
        # Save and analyze
        if not args.no_save:
            mixed_bytes = mixed_audio.tobytes()
            save_as_wav(mixed_bytes, final_sample_rate, 1, args.output)
        
        recording_float = mixed_audio.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(recording_float**2))
        peak = np.max(np.abs(recording_float))
        
        print(f"\n--- Audio Analysis ---")
        print(f"Duration: {len(recording_float) / final_sample_rate:.2f}s")
        print(f"Sample rate: {final_sample_rate} Hz")
        print(f"Peak: {peak:.4f}, RMS: {rms:.6f}")
        
        if rms < 0.001:
            print("Result: Silent/quiet recording.")
        else:
            level = 'Low' if rms < 0.01 else 'Medium' if rms < 0.1 else 'High'
            print(f"Result: Success! Audio level: {level}")

    finally:
        if p:
            p.terminate()


if __name__ == "__main__":
    main() 
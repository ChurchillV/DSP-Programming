import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_complex_sinusoids():
    """
    Plot various representations of complex sinusoids
    """
    # Time parameters
    t = np.linspace(0, 4*np.pi, 1000)
    
    # Complex sinusoid parameters
    frequencies = [1, 2, 0.5]
    amplitudes = [1, 1.5, 0.8]
    phases = [0, np.pi/4, np.pi/2]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Real and Imaginary parts separately
    ax1 = fig.add_subplot(3, 3, 1)
    for i, (freq, amp, phase) in enumerate(zip(frequencies, amplitudes, phases)):
        z = amp * np.exp(1j * (2*np.pi*freq*t + phase))
        ax1.plot(t, np.real(z), label=f'Re(z{i+1})', linestyle='-')
        ax1.plot(t, np.imag(z), label=f'Im(z{i+1})', linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Real and Imaginary Parts')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Magnitude and Phase
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    for i, (freq, amp, phase) in enumerate(zip(frequencies, amplitudes, phases)):
        z = amp * np.exp(1j * (2*np.pi*freq*t + phase))
        ax2.plot(t, np.abs(z), label=f'|z{i+1}|')
        ax3.plot(t, np.angle(z), label=f'∠z{i+1}')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Magnitude')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Phase (radians)')
    ax3.set_title('Phase')
    ax3.legend()
    ax3.grid(True)
    
    # 3. Complex plane representation (phasor diagram)
    ax4 = fig.add_subplot(3, 3, 4)
    colors = ['blue', 'red', 'green']
    for i, (freq, amp, phase) in enumerate(zip(frequencies, amplitudes, phases)):
        # Show multiple time instances
        time_instances = np.linspace(0, 2*np.pi/freq, 8)
        z_instances = amp * np.exp(1j * (2*np.pi*freq*time_instances + phase))
        
        ax4.plot(np.real(z_instances), np.imag(z_instances), 'o-', 
                color=colors[i], alpha=0.7, label=f'z{i+1}')
        
        # Draw arrows for the rotating phasor
        for j, z_val in enumerate(z_instances[::2]):  # Every other point
            ax4.annotate('', xy=(np.real(z_val), np.imag(z_val)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.5))
    
    ax4.set_xlabel('Real')
    ax4.set_ylabel('Imaginary')
    ax4.set_title('Complex Plane (Phasor Rotation)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    # 4. 3D helix representation
    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    freq, amp, phase = 1, 1, 0  # Use single sinusoid for clarity
    z = amp * np.exp(1j * (2*np.pi*freq*t + phase))
    ax5.plot(t, np.real(z), np.imag(z), 'b-', linewidth=2)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Real')
    ax5.set_zlabel('Imaginary')
    ax5.set_title('3D Helix Representation')
    
    # 5. Frequency domain representation
    ax6 = fig.add_subplot(3, 3, 6)
    # Create a signal with multiple frequencies
    fs = 100  # Sampling frequency
    t_fft = np.arange(0, 2, 1/fs)
    signal = (amplitudes[0] * np.exp(1j * (2*np.pi*frequencies[0]*t_fft + phases[0])) +
              amplitudes[1] * np.exp(1j * (2*np.pi*frequencies[1]*t_fft + phases[1])) +
              amplitudes[2] * np.exp(1j * (2*np.pi*frequencies[2]*t_fft + phases[2])))
    
    # FFT
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Plot positive frequencies only
    pos_mask = freqs >= 0
    ax6.stem(freqs[pos_mask], np.abs(fft_vals[pos_mask]), basefmt=' ')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude')
    ax6.set_title('Frequency Domain')
    ax6.grid(True)
    
    # 6. Superposition of complex sinusoids
    ax7 = fig.add_subplot(3, 3, 7)
    z_sum = np.zeros_like(t, dtype=complex)
    for i, (freq, amp, phase) in enumerate(zip(frequencies, amplitudes, phases)):
        z_i = amp * np.exp(1j * (2*np.pi*freq*t + phase))
        z_sum += z_i
        ax7.plot(t, np.real(z_i), alpha=0.5, linestyle='--', label=f'Re(z{i+1})')
    
    ax7.plot(t, np.real(z_sum), 'k-', linewidth=2, label='Re(sum)')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Real Part')
    ax7.set_title('Superposition (Real Parts)')
    ax7.legend()
    ax7.grid(True)
    
    # 7. Envelope and instantaneous frequency
    ax8 = fig.add_subplot(3, 3, 8)
    # Create AM modulated signal
    f_carrier = 5
    f_mod = 0.5
    mod_depth = 0.8
    envelope = 1 + mod_depth * np.cos(2*np.pi*f_mod*t)
    z_am = envelope * np.exp(1j * 2*np.pi*f_carrier*t)
    
    ax8.plot(t, np.real(z_am), 'b-', label='AM Signal')
    ax8.plot(t, envelope, 'r--', linewidth=2, label='Envelope')
    ax8.plot(t, -envelope, 'r--', linewidth=2)
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Amplitude')
    ax8.set_title('AM Modulated Complex Sinusoid')
    ax8.legend()
    ax8.grid(True)
    
    # 8. Waterfall plot (spectrogram-like)
    ax9 = fig.add_subplot(3, 3, 9)
    # Create chirp signal
    f0, f1 = 0.5, 3
    t_chirp = np.linspace(0, 2, 1000)
    chirp_rate = (f1 - f0) / t_chirp[-1]
    instantaneous_freq = f0 + chirp_rate * t_chirp
    phase_chirp = 2*np.pi * (f0*t_chirp + 0.5*chirp_rate*t_chirp**2)
    z_chirp = np.exp(1j * phase_chirp)
    
    ax9.plot(t_chirp, instantaneous_freq, 'r-', linewidth=2)
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Instantaneous Frequency')
    ax9.set_title('Chirp Signal Frequency')
    ax9.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_euler_relationship():
    """
    Demonstrate Euler's formula: e^(jθ) = cos(θ) + j*sin(θ)
    """
    theta = np.linspace(0, 4*np.pi, 1000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Euler's formula components
    euler_exp = np.exp(1j * theta)
    cos_component = np.cos(theta)
    sin_component = np.sin(theta)
    
    # Time domain
    ax1.plot(theta, cos_component, 'b-', label='cos(θ)')
    ax1.plot(theta, sin_component, 'r-', label='sin(θ)')
    ax1.plot(theta, np.real(euler_exp), 'b--', alpha=0.7, label='Re(e^(jθ))')
    ax1.plot(theta, np.imag(euler_exp), 'r--', alpha=0.7, label='Im(e^(jθ))')
    ax1.set_xlabel('θ (radians)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Euler\'s Formula Components')
    ax1.legend()
    ax1.grid(True)
    
    # Complex plane (unit circle)
    ax2.plot(np.real(euler_exp), np.imag(euler_exp), 'g-', linewidth=2, label='e^(jθ)')
    
    # Mark special points
    special_angles = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    special_points = np.exp(1j * np.array(special_angles))
    ax2.plot(np.real(special_points), np.imag(special_points), 'ro', markersize=8)
    
    # Add angle markers
    for i, angle in enumerate(special_angles[:-1]):
        point = special_points[i]
        ax2.annotate(f'{angle:.2f}', 
                    xy=(np.real(point), np.imag(point)),
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imaginary')
    ax2.set_title('Unit Circle in Complex Plane')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Plotting complex sinusoids...")
    plot_complex_sinusoids()
    print("\nDemonstrating Euler's relationship...")
    plot_euler_relationship()
    print("Done!")
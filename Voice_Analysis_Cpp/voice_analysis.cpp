#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>

// Audio recording (cross-platform)
#ifdef _WIN32
    #include <windows.h>
    #include <mmsystem.h>
    #pragma comment(lib, "winmm.lib")
#else
    #include <alsa/asoundlib.h>
#endif

// For plotting - using gnuplot interface
#include <cstdlib>
#include <sstream>

class VoiceRecorderPlotter {
private:
    int sampleRate;
    int channels;
    std::vector<double> recording;
    
    // FFT implementation
    void fft(std::vector<std::complex<double>>& data) {
        int n = data.size();
        if (n <= 1) return;
        
        // Divide
        std::vector<std::complex<double>> even, odd;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) even.push_back(data[i]);
            else odd.push_back(data[i]);
        }
        
        // Conquer
        fft(even);
        fft(odd);
        
        // Combine
        for (int i = 0; i < n/2; i++) {
            std::complex<double> t = std::polar(1.0, -2 * M_PI * i / n) * odd[i];
            data[i] = even[i] + t;
            data[i + n/2] = even[i] - t;
        }
    }
    
    // Calculate next power of 2
    int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) power <<= 1;
        return power;
    }
    
public:
    VoiceRecorderPlotter(int sr = 44100, int ch = 1) 
        : sampleRate(sr), channels(ch) {}
    
    // Record audio using platform-specific APIs
    bool recordAudio(double duration) {
        recording.clear();
        int numSamples = static_cast<int>(duration * sampleRate);
        recording.resize(numSamples);
        
        std::cout << "Recording for " << duration << " seconds... Speak now!" << std::endl;
        std::cout << "3... 2... 1... GO!" << std::endl;
        
#ifdef _WIN32
        return recordAudioWindows(duration, numSamples);
#else
        return recordAudioLinux(duration, numSamples);
#endif
    }
    
#ifdef _WIN32
    bool recordAudioWindows(double duration, int numSamples) {
        WAVEFORMATEX waveFormat;
        waveFormat.wFormatTag = WAVE_FORMAT_PCM;
        waveFormat.nChannels = channels;
        waveFormat.nSamplesPerSec = sampleRate;
        waveFormat.wBitsPerSample = 16;
        waveFormat.nBlockAlign = waveFormat.nChannels * waveFormat.wBitsPerSample / 8;
        waveFormat.nAvgBytesPerSec = waveFormat.nSamplesPerSec * waveFormat.nBlockAlign;
        waveFormat.cbSize = 0;
        
        HWAVEIN hWaveIn;
        MMRESULT result = waveInOpen(&hWaveIn, WAVE_MAPPER, &waveFormat, 0, 0, WAVE_FORMAT_DIRECT);
        
        if (result != MMSYSERR_NOERROR) {
            std::cerr << "Failed to open wave input device" << std::endl;
            return false;
        }
        
        // Allocate buffer
        int bufferSize = numSamples * sizeof(short);
        std::vector<short> buffer(numSamples);
        
        WAVEHDR waveHeader;
        waveHeader.lpData = reinterpret_cast<LPSTR>(buffer.data());
        waveHeader.dwBufferLength = bufferSize;
        waveHeader.dwBytesRecorded = 0;
        waveHeader.dwUser = 0;
        waveHeader.dwFlags = 0;
        waveHeader.dwLoops = 0;
        waveHeader.lpNext = 0;
        waveHeader.reserved = 0;
        
        waveInPrepareHeader(hWaveIn, &waveHeader, sizeof(WAVEHDR));
        waveInAddBuffer(hWaveIn, &waveHeader, sizeof(WAVEHDR));
        waveInStart(hWaveIn);
        
        // Wait for recording to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000)));
        
        waveInStop(hWaveIn);
        waveInUnprepareHeader(hWaveIn, &waveHeader, sizeof(WAVEHDR));
        waveInClose(hWaveIn);
        
        // Convert to double and normalize
        for (int i = 0; i < numSamples && i < buffer.size(); i++) {
            recording[i] = static_cast<double>(buffer[i]) / 32768.0;
        }
        
        return true;
    }
#else
    bool recordAudioLinux(double duration, int numSamples) {
        snd_pcm_t *handle;
        snd_pcm_hw_params_t *params;
        
        // Open PCM device for recording
        int rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
        if (rc < 0) {
            std::cerr << "Unable to open PCM device: " << snd_strerror(rc) << std::endl;
            return false;
        }
        
        // Allocate hardware parameters object
        snd_pcm_hw_params_alloca(&params);
        snd_pcm_hw_params_any(handle, params);
        
        // Set parameters
        snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);
        snd_pcm_hw_params_set_channels(handle, params, channels);
        
        unsigned int rate = sampleRate;
        snd_pcm_hw_params_set_rate_near(handle, params, &rate, 0);
        
        // Apply parameters
        rc = snd_pcm_hw_params(handle, params);
        if (rc < 0) {
            std::cerr << "Unable to set hw parameters: " << snd_strerror(rc) << std::endl;
            snd_pcm_close(handle);
            return false;
        }
        
        // Record audio
        std::vector<short> buffer(numSamples);
        snd_pcm_sframes_t frames = snd_pcm_readi(handle, buffer.data(), numSamples);
        
        if (frames < 0) {
            frames = snd_pcm_recover(handle, frames, 0);
        }
        
        snd_pcm_close(handle);
        
        // Convert to double and normalize
        for (int i = 0; i < numSamples && i < buffer.size(); i++) {
            recording[i] = static_cast<double>(buffer[i]) / 32768.0;
        }
        
        return true;
    }
#endif
    
    void plotWaveform() {
        if (recording.empty()) {
            std::cerr << "No recording data available" << std::endl;
            return;
        }
        
        // Write data to temporary file
        std::ofstream dataFile("waveform_data.txt");
        for (size_t i = 0; i < recording.size(); i++) {
            double time = static_cast<double>(i) / sampleRate;
            dataFile << time << " " << recording[i] << std::endl;
        }
        dataFile.close();
        
        // Create gnuplot script
        std::ofstream plotFile("plot_waveform.gp");
        plotFile << "set terminal png size 1200,400" << std::endl;
        plotFile << "set output 'waveform.png'" << std::endl;
        plotFile << "set title 'Voice Recording - Time Domain'" << std::endl;
        plotFile << "set xlabel 'Time (seconds)'" << std::endl;
        plotFile << "set ylabel 'Amplitude'" << std::endl;
        plotFile << "set grid" << std::endl;
        plotFile << "plot 'waveform_data.txt' using 1:2 with lines title 'Waveform'" << std::endl;
        plotFile.close();
        
        system("gnuplot plot_waveform.gp");
        std::cout << "Waveform saved as waveform.png" << std::endl;
    }
    
    void plotFrequencySpectrum() {
        if (recording.empty()) {
            std::cerr << "No recording data available" << std::endl;
            return;
        }
        
        // Prepare data for FFT
        int fftSize = nextPowerOf2(recording.size());
        std::vector<std::complex<double>> fftData(fftSize);
        
        for (size_t i = 0; i < recording.size(); i++) {
            fftData[i] = std::complex<double>(recording[i], 0.0);
        }
        
        // Perform FFT
        fft(fftData);
        
        // Write frequency spectrum data
        std::ofstream dataFile("spectrum_data.txt");
        for (int i = 0; i < fftSize / 2; i++) {
            double frequency = static_cast<double>(i * sampleRate) / fftSize;
            double magnitude = std::abs(fftData[i]);
            if (frequency <= 4000) { // Focus on speech range
                dataFile << frequency << " " << magnitude << std::endl;
            }
        }
        dataFile.close();
        
        // Create gnuplot script
        std::ofstream plotFile("plot_spectrum.gp");
        plotFile << "set terminal png size 1200,600" << std::endl;
        plotFile << "set output 'spectrum.png'" << std::endl;
        plotFile << "set title 'Voice Recording - Frequency Spectrum'" << std::endl;
        plotFile << "set xlabel 'Frequency (Hz)'" << std::endl;
        plotFile << "set ylabel 'Magnitude'" << std::endl;
        plotFile << "set grid" << std::endl;
        plotFile << "set logscale y" << std::endl;
        plotFile << "plot 'spectrum_data.txt' using 1:2 with lines title 'Spectrum'" << std::endl;
        plotFile.close();
        
        system("gnuplot plot_spectrum.gp");
        std::cout << "Frequency spectrum saved as spectrum.png" << std::endl;
    }
    
    void calculateEnvelope() {
        if (recording.empty()) return;
        
        std::vector<double> envelope;
        int windowSize = sampleRate / 100; // 10ms window
        
        for (size_t i = 0; i < recording.size(); i += windowSize/2) {
            double maxVal = 0.0;
            for (int j = 0; j < windowSize && (i + j) < recording.size(); j++) {
                maxVal = std::max(maxVal, std::abs(recording[i + j]));
            }
            envelope.push_back(maxVal);
        }
        
        // Write envelope data
        std::ofstream dataFile("envelope_data.txt");
        std::ofstream waveFile("wave_envelope_data.txt");
        
        for (size_t i = 0; i < recording.size(); i++) {
            double time = static_cast<double>(i) / sampleRate;
            waveFile << time << " " << recording[i] << std::endl;
        }
        
        for (size_t i = 0; i < envelope.size(); i++) {
            double time = static_cast<double>(i * windowSize/2) / sampleRate;
            dataFile << time << " " << envelope[i] << std::endl;
            dataFile << time << " " << -envelope[i] << std::endl;
        }
        
        dataFile.close();
        waveFile.close();
        
        // Create gnuplot script
        std::ofstream plotFile("plot_envelope.gp");
        plotFile << "set terminal png size 1200,400" << std::endl;
        plotFile << "set output 'envelope.png'" << std::endl;
        plotFile << "set title 'Voice Recording - Amplitude Envelope'" << std::endl;
        plotFile << "set xlabel 'Time (seconds)'" << std::endl;
        plotFile << "set ylabel 'Amplitude'" << std::endl;
        plotFile << "set grid" << std::endl;
        plotFile << "plot 'wave_envelope_data.txt' using 1:2 with lines title 'Waveform' lc rgb 'blue', \\" << std::endl;
        plotFile << "     'envelope_data.txt' using 1:2 with lines title 'Envelope' lc rgb 'red' lw 2" << std::endl;
        plotFile.close();
        
        system("gnuplot plot_envelope.gp");
        std::cout << "Envelope plot saved as envelope.png" << std::endl;
    }
    
    void analyzeVoice() {
        if (recording.empty()) {
            std::cerr << "No recording data available" << std::endl;
            return;
        }
        
        // Calculate statistics
        double maxAmplitude = *std::max_element(recording.begin(), recording.end(), 
            [](double a, double b) { return std::abs(a) < std::abs(b); });
        
        double rmsAmplitude = 0.0;
        for (double sample : recording) {
            rmsAmplitude += sample * sample;
        }
        rmsAmplitude = std::sqrt(rmsAmplitude / recording.size());
        
        double duration = static_cast<double>(recording.size()) / sampleRate;
        
        std::cout << "\n=== Voice Analysis Results ===" << std::endl;
        std::cout << "Duration: " << std::fixed << std::setprecision(2) << duration << " seconds" << std::endl;
        std::cout << "Sample Rate: " << sampleRate << " Hz" << std::endl;
        std::cout << "Max Amplitude: " << std::fixed << std::setprecision(3) << std::abs(maxAmplitude) << std::endl;
        std::cout << "RMS Amplitude: " << std::fixed << std::setprecision(3) << rmsAmplitude << std::endl;
        std::cout << "Total Samples: " << recording.size() << std::endl;
    }
    
    void comprehensiveAnalysis(double duration = 5.0) {
        std::cout << "Voice Recording and Analysis Tool (C++)" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Record audio
        if (!recordAudio(duration)) {
            std::cerr << "Failed to record audio" << std::endl;
            return;
        }
        
        std::cout << "Recording completed successfully!" << std::endl;
        
        // Normalize audio
        if (!recording.empty()) {
            double maxVal = *std::max_element(recording.begin(), recording.end(),
                [](double a, double b) { return std::abs(a) < std::abs(b); });
            if (std::abs(maxVal) > 0) {
                for (double& sample : recording) {
                    sample /= std::abs(maxVal);
                }
            }
        }
        
        // Analyze and plot
        analyzeVoice();
        
        std::cout << "\nGenerating plots..." << std::endl;
        plotWaveform();
        plotFrequencySpectrum();
        calculateEnvelope();
        
        std::cout << "\nAll plots generated successfully!" << std::endl;
        std::cout << "Files created: waveform.png, spectrum.png, envelope.png" << std::endl;
    }
};

int main() {
    try {
        VoiceRecorderPlotter recorder(44100, 1);
        
        std::cout << "Enter recording duration in seconds (default 5): ";
        std::string input;
        std::getline(std::cin, input);
        
        double duration = 5.0;
        if (!input.empty()) {
            try {
                duration = std::stod(input);
            } catch (const std::exception& e) {
                std::cout << "Invalid input, using default duration of 5 seconds" << std::endl;
            }
        }
        
        recorder.comprehensiveAnalysis(duration);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
import { Audio } from 'expo-av';

export const requestAudioPermissions = async () => {
    const { status } = await Audio.requestPermissionsAsync();
    if (status === 'granted') {
        await Audio.setAudioModeAsync({
            allowsRecordingIOS: true,
            playsInSilentModeIOS: true,
            staysActiveInBackground: true,
        });
    }
    return status === 'granted';
};

export const startAudioStream = async (callback) => {
    try {
        const recording = new Audio.Recording();
        await recording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);

        recording.setOnRecordingStatusUpdate(status => {
            if (callback) {
                // Normalize metering to 0-1 range roughly, or just pass db value
                // metering is usually -160 to 0 dB
                const metering = status.metering || -160;
                callback(metering);
            }
        });

        await recording.startAsync();
        return recording;
    } catch (error) {
        console.error('Failed to start recording', error);
        return null;
    }
};

export const stopAudioStream = async (recording) => {
    if (recording) {
        await recording.stopAndUnloadAsync();
    }
};

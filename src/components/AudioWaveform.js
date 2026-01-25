import React from 'react';
import { View, StyleSheet, Animated } from 'react-native';

export default function AudioWaveform({ metering }) {
    // metering is usually logarithmic, e.g. -160 to 0.
    // We want to map it to a height.
    // if metering is -160, height is roughly 0.
    // if metering is 0, height is roughly 100%.

    // Simple visualization: pulsating bar or just a bar graph
    // Let's do a simple bar for confirmation.

    const height = Math.max(0, (metering + 60) * 2); // Quick normalization logic, adjust as needed

    return (
        <View style={styles.container}>
            <View style={[styles.bar, { height: height, backgroundColor: height > 50 ? '#4caf50' : '#ffa726' }]} />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        height: 100,
        width: '100%',
        backgroundColor: '#eee',
        justifyContent: 'flex-end',
        alignItems: 'center',
        borderRadius: 8,
        overflow: 'hidden',
    },
    bar: {
        width: '100%',
        minHeight: 2,
    },
});

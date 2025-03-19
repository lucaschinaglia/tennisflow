import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView } from 'react-native';
import { Button } from '../components/ui/Button';
import { ChevronLeft, Camera } from '../components/ui/Icons';

type RecordVideoProps = {
  onBack?: () => void;
};

export default function RecordVideo({ onBack }: RecordVideoProps) {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        {onBack && (
          <Button
            variant="ghost"
            size="icon"
            style={styles.backButton}
            onPress={onBack}
          >
            <ChevronLeft size={20} color="#a5b4fc" />
          </Button>
        )}
        <Text style={styles.title}>Record Session</Text>
      </View>

      <View style={styles.cameraPlaceholder}>
        <Camera size={48} color="#64748b" />
        <Text style={styles.placeholderText}>Camera Preview</Text>
      </View>

      <View style={styles.controls}>
        <View style={styles.instructionCard}>
          <Text style={styles.instructionTitle}>Recording Instructions</Text>
          <Text style={styles.instructionText}>
            1. Place your device where it can capture your full motion
          </Text>
          <Text style={styles.instructionText}>
            2. Ensure good lighting conditions
          </Text>
          <Text style={styles.instructionText}>
            3. Perform your tennis strokes clearly
          </Text>
          <Text style={styles.instructionText}>
            4. Try to stay within the marked area while playing
          </Text>
        </View>

        <TouchableOpacity style={styles.recordButton}>
          <View style={styles.recordButtonInner} />
        </TouchableOpacity>

        <Text style={styles.helpText}>
          Tap the button to start recording your session
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a', // slate-950
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  cameraPlaceholder: {
    flex: 1,
    backgroundColor: '#1e293b', // slate-800
    justifyContent: 'center',
    alignItems: 'center',
    margin: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#334155', // slate-700
    borderStyle: 'dashed',
  },
  placeholderText: {
    marginTop: 12,
    color: '#64748b', // slate-500
    fontSize: 16,
  },
  controls: {
    padding: 16,
    alignItems: 'center',
  },
  instructionCard: {
    backgroundColor: '#1e293b', // slate-800
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    width: '100%',
    borderWidth: 1,
    borderColor: '#334155', // slate-700
  },
  instructionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  instructionText: {
    color: '#cbd5e1', // slate-300
    marginBottom: 8,
    fontSize: 14,
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(239, 68, 68, 0.2)', // red-500/20
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  recordButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#ef4444', // red-500
  },
  helpText: {
    color: '#94a3b8', // slate-400
    textAlign: 'center',
    fontSize: 14,
  },
}); 
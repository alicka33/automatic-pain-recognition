import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, ActivityIndicator, TouchableOpacity, Text, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video } from 'expo-av';
import { useColorScheme } from 'react-native';
import { router } from 'expo-router';
import { ThemedText } from '@/components/ThemedText';
import { PainColors, Colors } from '@/constants/Colors';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback } from 'react';
import { uploadAndAnalyzeVideo } from '@/services/videoAnalysis';

export default function CameraScreen() {
  const videoRef = useRef<Video>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [videoAspect, setVideoAspect] = useState(1);
  const [loading, setLoading] = useState(false);
  const [isStartingUp, setIsStartingUp] = useState(true);
  const colorScheme = useColorScheme() ?? 'light';
  const tintColor = Colors[colorScheme].tint;

  useFocusEffect(
    useCallback(() => {
      const startUp = async () => {
        setIsStartingUp(true);
        await handleRecord();
        setIsStartingUp(false);
      };
      startUp();

      // Cleanup when navigating away from screen
      return () => {
        console.log("Cleaning up after screen navigation...");
        cleanupVideoResources();
        setVideoUri(null);
        setVerdict(null);
        setVideoAspect(1);
        if (videoRef.current) videoRef.current = null;
      };
    }, [])
  );

  const cleanup = async () => {
    if (videoRef.current) {
      await videoRef.current.setStatusAsync({ shouldPlay: false }).catch(() => {});
      await videoRef.current.unloadAsync().catch(() => {});
    }
    setVideoUri(null);
    setVerdict(null);
    setVideoAspect(1);
  };

  const cleanupVideoResources = async () => {
    console.log("Cleanup: releasing video resources...");
    try {
      if (videoRef.current) {
        await videoRef.current.setStatusAsync({ shouldPlay: false }).catch(() => {});
        await videoRef.current.unloadAsync().catch(() => {});
        videoRef.current = null;
      }
    } catch (err) {
      console.warn("Error releasing video resources:", err);
    }
  };

  const handleRecord = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      console.error('No camera permissions.');
      router.back();
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      videoQuality: ImagePicker.UIImagePickerPresentationStyle.FullScreen,
    });

    if (!result.canceled && result.assets.length > 0) {
      await startAnalysis(result.assets[0].uri);
    } else {
      router.back();
    }
  };

  const startAnalysis = async (uri: string) => {
    setLoading(true);
    setVideoUri(uri);
    try {
      const result = await uploadAndAnalyzeVideo(uri);
      setVerdict(result);
    } catch (err) {
      console.error(err);
      setVerdict('UNKNOWN');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoReady = (status) => {
    if (status.naturalSize?.width && status.naturalSize?.height)
      setVideoAspect(status.naturalSize.width / status.naturalSize.height);
  };

  const currentPain = verdict ? PainColors[verdict] : null;

  if (isStartingUp)
    return <View style={styles.centered}><ActivityIndicator size="large" color={tintColor} /><Text style={styles.loadingText}>Loading...</Text></View>;

  if (loading)
    return <View style={styles.centered}><ActivityIndicator size="large" color={tintColor} /><Text style={styles.loadingText}>Analyzing video...</Text></View>;

  if (videoUri && verdict)
    return (
      <View style={styles.container}>
        <View style={styles.videoAndVerdictContainer}>
          <View style={[styles.videoFrame, { borderColor: currentPain?.color || tintColor, aspectRatio: videoAspect }]}>
            <Video ref={videoRef} source={{ uri: videoUri }} useNativeControls resizeMode="cover" isLooping shouldPlay onReadyForDisplay={handleVideoReady} style={styles.videoPlayer} />
          </View>
          <View style={styles.verdictCardContainer}>
            <View style={[styles.verdictCard, { borderColor: currentPain?.color || tintColor }]}>
              <Text style={styles.verdictTitle}>Analysis result</Text>
              <Text style={[styles.verdictValue, { color: currentPain?.color || tintColor }]}>{currentPain?.description || 'Unknown pain level'}</Text>
            </View>
          </View>
        </View>
        <TouchableOpacity
          style={[styles.secondaryButton, { borderColor: tintColor }]}
          onPress={async () => {
            await cleanupVideoResources();
            setVideoUri(null);
            setVerdict(null);
            setVideoAspect(1);
            router.replace('/');
          }}
        >
          <Text style={[styles.secondaryButtonText, { color: tintColor }]}>
            Return to menu
          </Text>
        </TouchableOpacity>
      </View>
    );

  return (
    <View style={styles.centered}>
      <ThemedText>An error occurred.</ThemedText>
      <TouchableOpacity onPress={() => router.replace('/(pages)/')}><Text style={{ color: tintColor }}>Return</Text></TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, paddingTop: Platform.OS === 'ios' ? 50 : 30, backgroundColor: Colors.light.background, alignItems: 'center' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: Colors.light.background },
  loadingText: { marginTop: 15, fontSize: 16, color: '#777' },
  videoAndVerdictContainer: { width: '100%', flex: 1, alignItems: 'center' },
  videoFrame: { width: '100%', maxWidth: 350, flex: 1, borderWidth: 2, borderRadius: 12, overflow: 'hidden', marginBottom: 20, backgroundColor: 'black', justifyContent: 'center', alignItems: 'center' },
  videoPlayer: { flex: 1, width: '100%' },
  verdictCardContainer: { width: '100%', alignItems: 'center', marginBottom: 10 },
  verdictCard: { width: '90%', backgroundColor: 'rgba(255,255,255,0.08)', borderWidth: 2, borderRadius: 16, paddingVertical: 18, paddingHorizontal: 20, alignItems: 'center' },
  verdictTitle: { fontSize: 15, fontWeight: '500', color: '#bbb', marginBottom: 6, textTransform: 'uppercase' },
  verdictValue: { fontSize: 22, fontWeight: '700', textAlign: 'center' },
  secondaryButton: { width: '90%', paddingVertical: 13, borderRadius: 12, borderWidth: 1.5, alignItems: 'center', justifyContent: 'center', marginBottom: 30 },
  secondaryButtonText: { fontSize: 15, fontWeight: '600' },
});

import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, Button, ActivityIndicator, Text, TouchableOpacity, Platform, Alert } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { router, useFocusEffect } from 'expo-router';
import { PainColors, Colors } from '@/constants/Colors';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useColorScheme } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { uploadAndAnalyzeVideo } from '@/services/videoAnalysis';

const ANALYSIS_INTERVAL_MS = 5500;  // Same video length was used for model training

export default function VideoAnalyzerScreen() {
    const colorScheme = useColorScheme() ?? 'light';
    const tintColor = Colors[colorScheme].tint;

    const cameraRef = useRef<CameraView>(null);
    const [permission, requestPermission] = useCameraPermissions();
    const [isCameraReady, setIsCameraReady] = useState(false);
    const [facing, setFacing] = useState<'front' | 'back'>('front');
    const [cameraFacing, setCameraFacing] = useState<'front' | 'back'>('front');

    const [verdict, setVerdict] = useState<string | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisError, setAnalysisError] = useState<string | null>(null);

    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    const handleCameraReady = () => {
        setIsCameraReady(true);
        console.log("Camera is ready for recording.");
    };

    const toggleCameraFacing = () => {
      setFacing((prev) => (prev === 'front' ? 'back' : 'front'));
    };

    useEffect(() => {
        if (!permission || !permission.granted) {
            requestPermission();
        }
    }, [permission]);

    const processSegment = async () => {
        if (!cameraRef.current) return;

        // Stop recording (if still running) and get URI
        let video: any = null;
        try {
            console.log("Stopping recording...");
            await new Promise<void>((resolve, reject) => {
                if (!cameraRef.current) return reject("Missing camera reference");
                (cameraRef.current as any).stopRecording();
                // Wait a moment for expo-camera to save the file
                setTimeout(() => resolve(), 500);
            });
        } catch (e) {
            console.error("Error stopping recording:", e);
        }

        try {
            console.log("Starting new recording...");
            video = await (cameraRef.current as any).recordAsync({
                quality: "480p",
                maxDuration: ANALYSIS_INTERVAL_MS / 1000,
            });
            console.log("Recording completed:", video.uri);
        } catch (e) {
            console.error("Error in recordAsync:", e);
        }


        if (!video || !video.uri) {
            console.error("Error: Failed to get URI from recording. Attempting to resume recording...");
            if (cameraRef.current && (cameraRef.current as any).recordAsync) {
                (cameraRef.current as any).recordAsync({
                    quality: '480p',
                    maxDuration: ANALYSIS_INTERVAL_MS / 1000,
                }).catch((e: Error) => console.error("Error in emergency recording resume:", e));
            }
            return;
        }

        console.log(`Segment recording completed: ${video.uri}`);

        if (cameraRef.current && (cameraRef.current as any).recordAsync) {
            (cameraRef.current as any).recordAsync({
                quality: '480p',
                maxDuration: ANALYSIS_INTERVAL_MS / 1000,
            }).catch((e: Error) => console.error("Error resuming recording:", e));
        }

        setIsAnalyzing(true);
        try {
            const painLevel = await uploadAndAnalyzeVideo(video.uri);
            setVerdict(painLevel);
            setAnalysisError(null);
            console.log(`Received verdict: ${painLevel}`);

        } catch (error) {
            Alert.alert("Analysis Error", (error as Error).message || "Unknown error occurred while uploading video. Check token and connection.");
            setAnalysisError("Segment analysis error. See console.");
            console.error("Segment analysis error:", error);
        } finally {
            setIsAnalyzing(false);
        }
    };


    const startRealTimeRecording = async () => {
        if (!permission || !permission.granted) {
            Alert.alert("No Permissions", "Camera permissions are required.");
            setAnalysisError("No camera permissions.");
            return;
        }

        if (!isCameraReady) {
             Alert.alert("Error", "Camera is not ready. Wait until the loading indicator disappears.");
             setAnalysisError("Camera is not ready.");
             return;
        }

        setIsRecording(true);
        setVerdict(null);
        setAnalysisError(null);

        if (cameraRef.current && (cameraRef.current as any).recordAsync) {
            console.log("Starting first recording segment...");
            (cameraRef.current as any).recordAsync({
                quality: '480p',
                maxDuration: ANALYSIS_INTERVAL_MS / 1000,
            }).catch((e: Error) => console.error("Error starting recording:", e));
        } else {
            setAnalysisError("Camera is not ready or recordAsync method is missing.");
            setIsRecording(false);
            return;
        }

        intervalRef.current = setInterval(() => {
            processSegment();
        }, ANALYSIS_INTERVAL_MS);
    };

    const stopRealTimeRecording = async () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        if (cameraRef.current && (cameraRef.current as any).stopRecording) {
            try {
                await (cameraRef.current as any).stopRecording();
            } catch(e) {
                console.warn("Problem stopping recording.", e);
            }
        }
        setIsRecording(false);
        setVerdict(null);
    };

    useEffect(() => {
        return () => {
            stopRealTimeRecording();
        };
    }, []);

    useFocusEffect(
        React.useCallback(() => {
            return () => {
                if (isRecording) {
                    console.log("Screen lost focus - stopping recording");
                    stopRealTimeRecording();
                }
            };
        }, [isRecording])
    );


    if (!permission) {
        return <View style={styles.centeredContainer}><ActivityIndicator size="large" color={tintColor} /></View>;
    }
    if (!permission.granted) {
        return (
            <View style={styles.centeredContainer}>
                <ThemedText>No access to camera.</ThemedText>
                <Button title="Turn on Camera" onPress={requestPermission} color={tintColor} />
            </View>
        );
    }


    const currentPainData = verdict ? PainColors[verdict as keyof typeof PainColors] : null;
    const frameColor = currentPainData?.color || (isRecording ? '#FFA500' : tintColor);

    return (
        <View style={styles.container}>
            <View style={[
                styles.cameraContainer,
                { borderColor: frameColor, borderWidth: isRecording ? 4 : 1.5 }
            ]}>
                {console.log('Rendering CameraView')}
                <CameraView
                  ref={cameraRef}
                  style={styles.cameraPreview}
                  facing={cameraFacing}
                  mode="video"
                  enableTorch={false}
                  onCameraReady={() => {
                    console.log('Camera ready!');
                    setIsCameraReady(true);
                  }}
                />

                <TouchableOpacity
                  style={styles.switchCameraButton}
                  onPress={() => setCameraFacing(prev => prev === 'front' ? 'back' : 'front')}
                >
                  <Ionicons name="camera-reverse-outline" size={28} color="white" />
                </TouchableOpacity>
            </View>

            <View style={styles.verdictCardContainer}>
              <View style={[styles.verdictCard, { borderColor: frameColor }]}>
                <Text style={styles.verdictTitle}>Analysis result</Text>
                <Text
                  style={[
                    styles.verdictValue,
                    { color: currentPainData?.color || tintColor }
                  ]}
                >
                  {isRecording && !verdict
                    ? 'Recording...'
                    : isRecording
                    ? (currentPainData?.description || 'Waiting for result...')
                    : 'Analysis stopped'}
                </Text>
                {analysisError && (
                  <Text style={styles.errorText}>⚠️ {analysisError}</Text>
                )}
              </View>
            </View>

            <View style={styles.buttonContainer}>
              <TouchableOpacity
                style={[styles.actionButton, { backgroundColor: isRecording ? '#e74c3c' : '#27ae60' }]}
                onPress={isRecording ? stopRealTimeRecording : startRealTimeRecording}
                disabled={!isCameraReady || isAnalyzing}
              >
                <Text style={styles.actionButtonText}>
                  {isRecording ? 'Stop analysis' : 'Start real-time analysis'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.secondaryButton, { borderColor: tintColor }]}
                onPress={() => router.replace('/')}
              >
                <Text style={[styles.secondaryButtonText, { color: tintColor }]}>
                  Return to menu
                </Text>
              </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        padding: 20,
        paddingTop: Platform.OS === 'ios' ? 50 : 30,
        backgroundColor: Colors.light.background,
        alignItems: 'center',
    },
    centeredContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    cameraContainer: {
        width: '100%',
        flex: 1,
        borderRadius: 12,
        overflow: 'hidden',
        marginBottom: 20,
        backgroundColor: 'black',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 5,
        elevation: 8,
    },
    cameraPreview: {
        flex: 1,
        width: '100%',
    },
    overlay: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 10,
    },
    verdictBox: {
        width: '100%',
        padding: 15,
        borderRadius: 10,
        marginBottom: 10,
        alignItems: 'center',
        justifyContent: 'center',
    },
    buttonRow: {
        width: '100%',
        marginTop: 10,
    },
    switchCameraButton: {
      position: 'absolute',
      bottom: 20,
      right: 20,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      borderRadius: 30,
      width: 60,
      height: 60,
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 20,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 3 },
      shadowOpacity: 0.3,
      shadowRadius: 4,
      elevation: 6,
    },

    verdictCardContainer: {
      width: '100%',
      alignItems: 'center',
      marginBottom: 10,
    },

    verdictCard: {
      width: '90%',
      backgroundColor: 'rgba(255, 255, 255, 0.08)',
      borderWidth: 2,
      borderRadius: 16,
      paddingVertical: 18,
      paddingHorizontal: 20,
      alignItems: 'center',
      justifyContent: 'center',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.25,
      shadowRadius: 6,
      elevation: 6,
    },

    verdictTitle: {
      fontSize: 15,
      fontWeight: '500',
      letterSpacing: 0.4,
      color: '#bbb',
      marginBottom: 6,
      textTransform: 'uppercase',
    },

    verdictValue: {
      fontSize: 22,
      fontWeight: '700',
      textAlign: 'center',
      letterSpacing: 0.6,
    },

    errorText: {
      color: '#ff7675',
      marginTop: 6,
      fontSize: 13,
      fontWeight: '500',
    },

    buttonContainer: {
      width: '100%',
      alignItems: 'center',
      marginTop: 10,
      gap: 10,
    },

    actionButton: {
      width: '90%',
      paddingVertical: 14,
      borderRadius: 12,
      alignItems: 'center',
      justifyContent: 'center',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 3 },
      shadowOpacity: 0.25,
      shadowRadius: 5,
      elevation: 4,
    },

    actionButtonText: {
      color: 'white',
      fontSize: 16,
      fontWeight: '600',
      letterSpacing: 0.3,
    },

    secondaryButton: {
      width: '90%',
      paddingVertical: 13,
      borderRadius: 12,
      borderWidth: 1.5,
      alignItems: 'center',
      justifyContent: 'center',
    },

    secondaryButtonText: {
      fontSize: 15,
      fontWeight: '600',
      letterSpacing: 0.3,
    }
});

import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import 'react-native-reanimated';
import { View, StyleSheet, useColorScheme } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { Colors } from '@/constants/Colors';


SplashScreen.preventAutoHideAsync();

function LoadingScreen({ colorScheme }) {
  const color = Colors[colorScheme ?? 'light'].redCross;

  return (
    <View style={[styles.loadingContainer, { backgroundColor: Colors[colorScheme ?? 'light'].background }]}>
      <View style={[styles.crossContainer, { borderColor: color }]}>
        <View style={[styles.bar, { backgroundColor: color, width: '60%', height: 20 }]} />
        <View style={[styles.bar, { backgroundColor: color, height: '60%', width: 20, position: 'absolute' }]} />
      </View>
      <ThemedText type='title' style={styles.loadingText}>
        Pain Recognizer
      </ThemedText>
    </View>
  );
}

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const [fontLoaded] = useFonts({
    SpaceMono: require('@/assets/fonts/SpaceMono-Regular.ttf'),
  });

  const [appIsReady, setAppIsReady] = useState(false);

  useEffect(() => {
    if (fontLoaded) {
      const timeout = setTimeout(() => {
        setAppIsReady(true);
        SplashScreen.hideAsync();
      }, 2000);

      return () => clearTimeout(timeout);
    }
  }, [fontLoaded]);

  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
        <Stack>
          <Stack.Screen name="(pages)" options={{ headerShown: false }} />
        </Stack>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 30,
  },
  crossContainer: {
    width: 150,
    height: 150,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 75,
    borderWidth: 10,
  },
  bar: {
    borderRadius: 10,
  },
  loadingText: {
    fontSize: 40,
  }
});
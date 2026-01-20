import { ThemedText } from '@/components/ThemedText';
import { Link } from 'expo-router';
import React from 'react';
import { StyleSheet, View, TouchableOpacity, Text } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from 'react-native';

const MenuButton = ({ iconName, label, href }) => {
  const colorScheme = useColorScheme() ?? 'light';
  const color = Colors[colorScheme].tint;

  return (
    <Link href={href} asChild>
      <TouchableOpacity style={styles.button}>
        <View style={[styles.iconBox, { borderColor: color }]}>
          <Ionicons name={iconName} size={80} color={color} />
        </View>
        <ThemedText type='defaultSemiBold' style={{ color: color }}>
          {label}
        </ThemedText>
      </TouchableOpacity>
    </Link>
  );
};


export default function HomeScreen() {
  return (
    <View style={styles.pageContainer}>
      <ThemedText type='title' style={styles.title}>Choose an action</ThemedText>

      <View style={styles.menuContainer}>
        {/* Button 1: Record Video */}
        <MenuButton
          iconName="videocam-outline"
          label="Record Video"
          href={{ pathname: "/camera", params: { action: 'record' } }}
        />

        {/* Button 2: Choose from Library */}
        <MenuButton
          iconName="images-outline"
          label="Choose from library"
          href={{ pathname: "/library", params: { action: 'library' } }}
        />
        {/* Button 3: Real-Time Analysis */}
        <MenuButton
          iconName="pulse-outline"
          label="Real-Time Analysis"
          href="/record"
        />
      </View>
    </View>
  );
}


const styles = StyleSheet.create({
  pageContainer: {
    display: 'flex',
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
    gap: 40,
    backgroundColor: '#FFFFFF',
  },
  title: {
    fontSize: 28,
  },
  menuContainer: {
    display: 'flex',
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    gap: 20,
  },
  button: {
    alignItems: 'center',
    gap: 10,
    width: 150,
  },
  iconBox: {
    width: 150,
    height: 150,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 20,
    borderWidth: 2,
    borderStyle: 'dashed',
    padding: 10,
  }
});

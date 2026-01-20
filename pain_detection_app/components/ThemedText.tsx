import React from 'react';
import { Text, StyleSheet, useColorScheme } from 'react-native';
import { Colors } from '../constants/Colors';

export function ThemedText({ style, lightColor, darkColor, type = 'default', ...rest }) {
  const colorScheme = useColorScheme() ?? 'light';
  const color = lightColor && colorScheme === 'light' ? lightColor : darkColor && colorScheme === 'dark' ? darkColor : Colors[colorScheme].text;

  return <Text style={[
    { color },
    type === 'default' ? styles.default : undefined,
    type === 'title' ? styles.title : undefined,
    type === 'defaultSemiBold' ? styles.defaultSemiBold : undefined,
    style,
  ]} {...rest} />;
}

const styles = StyleSheet.create({
  default: {
    fontSize: 16,
    lineHeight: 24,
  },
  defaultSemiBold: {
    fontSize: 16,
    lineHeight: 24,
    fontWeight: '600',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    lineHeight: 36,
  },
});

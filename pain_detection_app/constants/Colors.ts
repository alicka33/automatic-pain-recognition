export const tintColorLight = '#D91E18';
export const tintColorDark = '#FF5252';

export const Colors = {
  light: {
    text: '#11181C',
    background: '#FFFFFF',
    tint: tintColorLight,
    icon: '#687076',
    tabIconDefault: '#687076',
    tabIconSelected: tintColorLight,
    redCross: tintColorLight,
  },
  dark: {
    text: '#11181C',
    background: '#151718',
    tint: tintColorDark,
    icon: '#9BA1A6',
    tabIconDefault: '#9BA1A6',
    tabIconSelected: tintColorDark,
    redCross: tintColorDark,
  },
};

export const PainColors: { [key: string]: { description: string, color: string } } = {
  "NO_PAIN": { color: '#5CB8E4', description: 'No Pain' },
  "WEAK_PAIN": { color: '#7EC8E3', description: 'Weak Pain' },
  "MODERATE_PAIN": { color: '#F4D35E', description: 'Moderate Pain' },
  "STRONG_PAIN": { color: '#F08A5D', description: 'Strong Pain' },
  "VERY_STRONG_PAIN": { color: '#C44536', description: 'Very Strong Pain' },

  "NO_FACE_DETECTED": { description: "No face detected", color: '#9AA5B1' },
  "UNKNOWN": { description: "Error", color: '#6B7280' },
};

export const PainLevels = Object.keys(PainColors);

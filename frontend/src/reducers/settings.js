/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'settings',
  initialState: {
    darkMode: false,
  },
  reducers: {
    toggleDarkMode: (state) => {
      state.darkMode = !state.darkMode;
    },
  },
});

export const { toggleDarkMode } = slice.actions;

export const darkModeSelector = (state) => state.settings.darkMode;

export default slice.reducer;

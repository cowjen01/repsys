/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const studioSlice = createSlice({
  name: 'studio',
  initialState: {
    buildMode: true,
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
  },
});

export const { toggleBuildMode } = studioSlice.actions;

export const buildModeSelector = (state) => state.studio.buildMode;

export default studioSlice.reducer;

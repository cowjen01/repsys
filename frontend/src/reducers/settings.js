/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'settings',
  initialState: {
    darkMode: false,
    itemView: {
      title: '',
      subtitle: '',
      caption: '',
      image: '',
      content: '',
    },
  },
  reducers: {
    setDarkMode: (state, { payload }) => {
      state.darkMode = payload;
    },
    setItemView: (state, { payload }) => {
      state.itemView = payload;
    },
  },
});

export const { setDarkMode, setItemView } = slice.actions;

export const darkModeSelector = (state) => state.settings.darkMode;
export const itemViewSelector = (state) => state.settings.itemView;

export default slice.reducer;

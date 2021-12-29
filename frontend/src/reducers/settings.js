/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'settings',
  initialState: {
    darkMode: false,
    itemFields: {
      title: null,
      image: null,
      content: null,
      caption: null,
      subtitle: null,
    },
  },
  reducers: {
    setDarkMode: (state, { payload }) => {
      state.darkMode = payload;
    },
    setItemFields: (state, { payload }) => {
      state.itemFields = payload;
    },
  },
});

export const { setDarkMode, setItemFields } = slice.actions;

export const darkModeSelector = (state) => state.settings.darkMode;
export const itemFieldsSelector = (state) => state.settings.itemFields;

export default slice.reducer;

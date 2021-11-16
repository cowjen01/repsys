import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

import { requestStateHandler } from './utils';

export const fetchInteractions = createAsyncThunk(
  'interactions/fetchInteractions',
  async (userId) => {
    const response = await axios.get('/api/interactions', {
      params: {
        user: userId,
      },
    });
    return response.data;
  }
);

export const slice = createSlice({
  name: 'interactions',
  initialState: {
    data: [],
    status: 'idle',
    error: null,
  },
  extraReducers: requestStateHandler(fetchInteractions),
});

export const interactionsSelector = (state) => state.interactions.data;
export const interactionsStatusSelector = (state) => state.interactions.status;

export default slice.reducer;

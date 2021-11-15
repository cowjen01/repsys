/* eslint-disable no-param-reassign */
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

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
    interactions: [],
    status: 'idle',
    error: null,
  },
  extraReducers(builder) {
    builder
      .addCase(fetchInteractions.pending, (state, action) => {
        state.status = 'loading';
      })
      .addCase(fetchInteractions.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.interactions = action.payload;
      })
      .addCase(fetchInteractions.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      });
  },
});

export const interactionsSelector = (state) => state.interactions.interactions;
export const interactionsStatusSelector = (state) => state.interactions.status;

export default slice.reducer;

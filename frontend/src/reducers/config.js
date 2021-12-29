import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

import { requestStateHandler } from './utils';

export const fetchConfig = createAsyncThunk('models/fetchConfig', async () => {
  const response = await axios.get('/api/config');
  return response.data;
});

export const slice = createSlice({
  name: 'config',
  initialState: {
    data: {
      models: [],
      dataset: null,
    },
    status: 'idle',
    error: null,
  },
  extraReducers: requestStateHandler(fetchConfig),
});

export const modelsSelector = (state) => state.config.data.models;
export const datasetSelector = (state) => state.config.data.dataset;
export const configStatusSelector = (state) => state.config.status;

export default slice.reducer;

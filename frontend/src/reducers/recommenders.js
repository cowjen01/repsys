/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

import { generateUID } from '../utils';

export const slice = createSlice({
  name: 'recommenders',
  initialState: [],
  reducers: {
    addRecommender: (state, action) => {
      state.push(action.payload);
    },
    setRecommenders: (state, action) => action.payload,
    updateRecommendersOrder: (state, action) => {
      const { dragIndex, hoverIndex } = action.payload;
      [state[dragIndex], state[hoverIndex]] = [state[hoverIndex], state[dragIndex]];
    },
    updateRecommender: (state, action) => {
      const { index, data } = action.payload;
      Object.assign(state[index], data);
    },
    duplicateRecommender: (state, action) => {
      const index = action.payload;
      const source = state[index];
      state.splice(index + 1, 0, {
        ...source,
        name: `${source.name} #${generateUID()}`,
      });
    },
    deleteRecommender: (state, action) => {
      state.splice(action.payload, 1);
    },
  },
});

export const {
  addRecommender,
  deleteRecommender,
  updateRecommendersOrder,
  duplicateRecommender,
  updateRecommender,
  setRecommenders,
} = slice.actions;

export const recommendersSelector = (state) => state.recommenders;

export const recommenderByIndexSelector = (index) => (state) => state.recommenders[index];

export default slice.reducer;

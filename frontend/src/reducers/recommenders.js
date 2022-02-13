/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'recommenders',
  initialState: [],
  reducers: {
    addRecommender: (state, action) => {
      state.push(action.payload);
    },
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
        name: `${source.name} - copy`,
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
} = slice.actions;

export const recommendersSelector = (state) => state.recommenders;

export default slice.reducer;

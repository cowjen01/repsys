import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const repsysApi = createApi({
  reducerPath: 'repsysApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api/' }),
  endpoints: (builder) => ({
    getUsers: builder.query({
      query: () => '/users',
    }),
    getItemsByTitle: builder.query({
      query: (title) => `/items?query=${title}`,
    }),
    getInteractionsByUser: builder.query({
      query: (id) => `/interactions?user=${id}`,
    }),
    getDataset: builder.query({
      query: () => '/dataset',
    }),
    getModels: builder.query({
      query: () => '/models',
    }),
    getRecomsForUser: builder.mutation({
      query: (body) => ({
        url: '/recommend',
        method: 'POST',
        body,
      }),
    }),
  }),
});

export const {
  useGetUsersQuery,
  useGetItemsByTitleQuery,
  useGetInteractionsByUserQuery,
  useGetDatasetQuery,
  useGetModelsQuery,
  useGetRecomsForUserMutation,
} = repsysApi;

import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const repsysApi = createApi({
  reducerPath: 'repsysApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api/' }),
  endpoints: (builder) => ({
    getUsers: builder.query({
      query: (split = 'train') => `/users/${split}`,
    }),
    getUsersByInteractions: builder.mutation({
      query: ({ split = 'train', attribute, value, threshold }) => ({
        url: `/users/${split}/search`,
        method: 'POST',
        body: {
          interactions: {
            attribute,
            value,
            threshold,
          },
        },
      }),
    }),
    getItemsByAttribute: builder.mutation({
      query: ({ attribute, value }) => ({
        url: `/items/search`,
        method: 'POST',
        body: {
          attribute,
          value,
        },
      }),
    }),
    getItemsByTitle: builder.query({
      query: (title) => `/items?query=${title}`,
    }),
    getUserByID: builder.query({
      query: (userID) => `/users/${userID}`,
    }),
    getUsersDescription: builder.mutation({
      query: ({ split = 'train', users }) => ({
        url: `/users/${split}/describe`,
        method: 'POST',
        body: users,
      }),
    }),
    getDataset: builder.query({
      query: () => '/dataset',
    }),
    getModels: builder.query({
      query: () => '/models',
    }),
    getMetricsByModel: builder.query({
      query: (model) => `/models/${model}/metrics`,
    }),
    getPredictionByModel: builder.mutation({
      query: ({ model, ...body }) => ({
        url: `/models/${model}/predict`,
        method: 'POST',
        body,
      }),
    }),
    getModelsMetrics: builder.query({
      query: () => '/models/metrics',
    }),
    getItemsDescription: builder.mutation({
      query: (items) => ({
        url: '/items/describe',
        method: 'POST',
        body: items,
      }),
    }),
    getUsersEmbeddings: builder.query({
      query: (split = 'train') => `/embeddings/${split}/users`,
    }),
    getItemsEmbeddings: builder.query({
      query: (split = 'train') => `/embeddings/${split}/items`,
    }),
  }),
});

export const {
  useGetDatasetQuery,
  useGetItemsByAttributeMutation,
  useGetItemsByTitleQuery,
  useGetItemsDescriptionMutation,
  useGetItemsEmbeddingsQuery,
  useGetMetricsByModelQuery,
  useGetModelsMetricsQuery,
  useGetModelsQuery,
  useGetPredictionByModelMutation,
  useGetUserByIDQuery,
  useGetUsersByInteractionsMutation,
  useGetUsersDescriptionMutation,
  useGetUsersEmbeddingsQuery,
  useGetUsersQuery
} = repsysApi;

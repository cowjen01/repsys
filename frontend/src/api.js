import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const repsysApi = createApi({
  reducerPath: 'repsysApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api/' }),
  endpoints: (builder) => ({
    getUsers: builder.query({
      query: ({ split = 'train' }) => `/users?split=${split}`,
    }),
    searchUsers: builder.mutation({
      query: ({ split = 'train', query }) => ({
        url: `/users/search`,
        method: 'POST',
        body: {
          split,
          query,
        },
      }),
    }),
    searchItems: builder.mutation({
      query: ({ query }) => ({
        url: `/items/search`,
        method: 'POST',
        body: {
          query,
        },
      }),
    }),
    getItemsByTitle: builder.query({
      query: (title) => `/items?query=${title}`,
    }),
    getUserByID: builder.query({
      query: (userID) => `/users/${userID}`,
    }),
    describeUsers: builder.mutation({
      query: ({ split = 'train', users }) => ({
        url: `/users/describe`,
        method: 'POST',
        body: {
          split,
          users,
        },
      }),
    }),
    getDataset: builder.query({
      query: () => '/dataset',
    }),
    getModels: builder.query({
      query: () => '/models',
    }),
    getUserMetricsByModel: builder.query({
      query: (model) => `/models/${model}/metrics/user`,
    }),
    getItemMetricsByModel: builder.query({
      query: (model) => `/models/${model}/metrics/item`,
    }),
    predictItemsByModel: builder.mutation({
      query: ({ model, ...body }) => ({
        url: `/models/${model}/predict`,
        method: 'POST',
        body,
      }),
    }),
    getModelsMetrics: builder.query({
      query: () => '/models/metrics',
    }),
    describeItems: builder.mutation({
      query: ({ items }) => ({
        url: '/items/describe',
        method: 'POST',
        body: {
          items,
        },
      }),
    }),
    getUserEmbeddings: builder.query({
      query: (split = 'train') => `/users/embeddings?split=${split}`,
    }),
    getItemEmbeddings: builder.query({
      query: (split = 'train') => `/items/embeddings?split=${split}`,
    }),
  }),
});

export const {
  useGetDatasetQuery,
  useGetItemsByTitleQuery,
  useGetItemEmbeddingsQuery,
  useGetUserMetricsByModelQuery,
  useGetModelsMetricsQuery,
  useGetModelsQuery,
  useGetUserByIDQuery,
  useGetUserEmbeddingsQuery,
  useGetUsersQuery,
  useDescribeItemsMutation,
  useDescribeUsersMutation,
  useSearchItemsMutation,
  useSearchUsersMutation,
  usePredictItemsByModelMutation,
  useGetItemMetricsByModelQuery,
} = repsysApi;

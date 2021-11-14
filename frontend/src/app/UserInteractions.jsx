import React from 'react';
import { Box, Paper } from '@mui/material';
import Skeleton from '@mui/material/Skeleton';
import Chip from '@mui/material/Chip';
import FilterListIcon from '@mui/icons-material/FilterList';
import { useSelector, useDispatch } from 'react-redux';
import { FixedSizeList } from 'react-window';

import { getRequest } from './api';
import ItemListView from './ItemListView';
import {
  clearCustomInteractions,
  customInteractionsSelector,
  selectedUserSelector,
} from '../reducers/studio';

function renderRow({ index, style, data }) {
  const item = data[index];
  return (
    <ItemListView
      image={item.image}
      key={item.id}
      id={item.id}
      title={item.title}
      subtitle={item.subtitle}
    />
  );
}

function UserInteractions() {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);

  const { items: userHistoryData, isLoading: isUserHistoryLoading } = getRequest('/interactions', {
    user: selectedUser ? selectedUser.id : null,
  });

  const handleDelete = () => {
    dispatch(clearCustomInteractions());
  };

  const interactions = customInteractions.length > 0 ? customInteractions : userHistoryData;

  return (
    <Box sx={{ marginTop: 2 }}>
      {customInteractions.length > 0 && (
        <Chip
          sx={{ marginBottom: 2 }}
          onDelete={handleDelete}
          icon={<FilterListIcon />}
          label="Custom interactions"
        />
      )}
      {!isUserHistoryLoading ? (
        <Paper>
          <FixedSizeList
            height={380}
            itemData={interactions}
            itemSize={100}
            itemCount={interactions.length}
            overscanCount={10}
          >
            {renderRow}
          </FixedSizeList>
        </Paper>
      ) : (
        <Skeleton variant="rectangular" height={380} width="100%" />
      )}
    </Box>
  );
}

export default UserInteractions;

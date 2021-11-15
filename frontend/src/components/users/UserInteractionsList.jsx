import React from 'react';
import { Box, Paper, Skeleton, Chip, Typography } from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import { useSelector, useDispatch } from 'react-redux';
import { FixedSizeList } from 'react-window';

import { getRequest } from '../../api';
import { ItemListView } from '../items';
import {
  setCustomInteractions,
  customInteractionsSelector,
  selectedUserSelector,
} from '../../reducers/root';

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

function UserInteractionsList() {
  const dispatch = useDispatch();
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);

  const { items: userHistoryData, isLoading: isUserHistoryLoading } = getRequest('/interactions', {
    user: selectedUser ? selectedUser.id : null,
  });

  const handleDelete = () => {
    dispatch(setCustomInteractions([]));
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
          <Typography
            sx={{
              paddingLeft: '16px',
              paddingRight: '16px',
              lineHeight: '48px',
              color: 'rgba(0, 0, 0, 0.6)',
            }}
            variant="subtitle2"
            component="div"
          >
            Interactions ({interactions.length})
          </Typography>
          <FixedSizeList
            height={330}
            itemData={interactions}
            itemSize={100}
            itemCount={interactions.length}
            overscanCount={15}
          >
            {renderRow}
          </FixedSizeList>
        </Paper>
      ) : (
        <Skeleton variant="rectangular" height={330} width="100%" />
      )}
    </Box>
  );
}

export default UserInteractionsList;

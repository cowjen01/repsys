import React from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import { closeItemDetailDialog, itemDetailDialogSelector } from '../../reducers/dialogs';
import { itemViewSelector } from '../../reducers/settings';

function ItemDetailDialog() {
  const dialog = useSelector(itemDetailDialogSelector);
  const dispatch = useDispatch();
  const itemView = useSelector(itemViewSelector);

  const handleClose = () => {
    dispatch(closeItemDetailDialog());
  };

  return (
    <Dialog open={dialog.open} maxWidth="sm" fullWidth onClose={handleClose}>
      {dialog.item && (
        <>
          <DialogTitle>{dialog.item[itemView.title]}</DialogTitle>
          <DialogContent>
            <Typography variant="subtitle1" component="p" gutterBottom>
              {[dialog.item[itemView.caption], dialog.item[itemView.subtitle]]
                .filter((x) => x !== undefined)
                .join(' | ')}
            </Typography>
            <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }} gutterBottom>
              {dialog.item[itemView.content] || 'No description provided.'}
            </Typography>
          </DialogContent>
        </>
      )}
      <DialogActions>
        <Button color="secondary" onClick={handleClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default ItemDetailDialog;

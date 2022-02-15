import React, { useMemo, useCallback } from 'react';
import { Formik, Field } from 'formik';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  Typography,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import {
  addRecommender,
  recommendersSelector,
  updateRecommender,
} from '../../reducers/recommenders';
import { recEditDialogSelector, closeRecEditDialog, openSnackbar } from '../../reducers/dialogs';
import { TextField, SelectField, CheckboxField } from '../fields';
import { useGetModelsQuery } from '../../api';
import { capitalize } from '../../utils';

function RecEditDialog() {
  const dialog = useSelector(recEditDialogSelector);
  const dispatch = useDispatch();
  const recommenders = useSelector(recommendersSelector);

  const models = useGetModelsQuery();

  const handleClose = () => {
    dispatch(closeRecEditDialog());
  };

  const handleSubmit = useCallback(
    (values) => {
      const data = {
        ...values,
        itemsPerPage: parseInt(values.itemsPerPage, 10),
      };

      // the index can be 0, so !dialog.index is not enough
      if (dialog.index === null) {
        dispatch(addRecommender(data));
      } else {
        dispatch(
          updateRecommender({
            index: dialog.index,
            data,
          })
        );
      }
      dispatch(
        openSnackbar({
          message: 'All settings applied!',
        })
      );
      handleClose();
    },
    [dispatch, dialog]
  );

  const initialValues = useMemo(() => {
    if (!models.isSuccess) {
      return null;
    }

    const defaultParams = Object.fromEntries(
      Object.entries(models.data).map(([modelName, modelData]) => [
        modelName,
        Object.fromEntries(
          Object.entries(modelData.params).map(([paramName, paramData]) => [
            paramName,
            paramData.default,
          ])
        ),
      ])
    );

    if (dialog.index === null) {
      return {
        name: 'New Recommender',
        itemsPerPage: 4,
        itemsLimit: 20,
        model: Object.keys(models.data)[0],
        modelParams: defaultParams,
      };
    }

    const data = recommenders[dialog.index];

    if (!defaultParams[data.model]) {
      return {
        ...data,
        model: '',
      };
    }

    return data;
  }, [dialog, models.isLoading]);

  return (
    <Dialog open={dialog.open} fullWidth maxWidth="sm" onClose={handleClose}>
      <DialogTitle>Recommender settings</DialogTitle>
      {!models.isLoading ? (
        <Formik
          initialValues={initialValues}
          validate={(values) => {
            const errors = {};
            const requiredMessage = 'This field is required.';
            if (!values.name) {
              errors.name = requiredMessage;
            }
            if (
              dialog.index === null &&
              recommenders.map(({ name }) => name).includes(values.name)
            ) {
              errors.name = 'The name must be unique.';
            }
            if (!values.itemsLimit) {
              errors.itemsLimit = requiredMessage;
            }
            if (!values.model) {
              errors.model = requiredMessage;
            }
            return errors;
          }}
          onSubmit={(values, { setSubmitting }) => {
            handleSubmit(values);
            setSubmitting(false);
          }}
        >
          {({ submitForm, isSubmitting, values }) => {
            const model = useMemo(() => models.data[values.model], [values.model]);
            return (
              <>
                <DialogContent>
                  <Grid container direction="column" spacing={2}>
                    <Grid item>
                      <Typography variant="subtitle2" component="div">
                        Appearance
                      </Typography>
                      <Field name="name" label="Name" fullWidth component={TextField} />
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Field
                            name="itemsPerPage"
                            label="Items per page"
                            component={SelectField}
                            options={[1, 3, 4].map((i) => ({ label: i, value: i }))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <Field
                            name="itemsLimit"
                            label="Max number of items"
                            component={TextField}
                            type="number"
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                    <Grid item>
                      <Typography variant="subtitle2" component="div">
                        Model configuration
                      </Typography>
                      <Field
                        name="model"
                        label="Model"
                        component={SelectField}
                        options={Object.keys(models.data).map((m) => ({ label: m, value: m }))}
                        displayEmpty
                      />
                      {model &&
                        model.params &&
                        Object.entries(model.params).map(([paramName, paramData]) => {
                          const name = `modelParams.${values.model}.${paramName}`;
                          const props = {
                            name,
                            label: capitalize(paramName),
                          };
                          if (paramData.field === 'select') {
                            return (
                              <Field
                                key={paramName}
                                component={SelectField}
                                options={paramData.options.map((b) => ({ label: b, value: b }))}
                                {...props}
                              />
                            );
                          }
                          if (paramData.field === 'checkbox') {
                            return <Field key={paramName} component={CheckboxField} {...props} />;
                          }
                          return (
                            <Field
                              key={paramName}
                              component={TextField}
                              type={paramData.field}
                              {...props}
                            />
                          );
                        })}
                    </Grid>
                  </Grid>
                </DialogContent>
                <DialogActions>
                  <Button onClick={handleClose}>Close</Button>
                  <Button onClick={submitForm} disabled={isSubmitting} autoFocus>
                    Save
                  </Button>
                </DialogActions>
              </>
            );
          }}
        </Formik>
      ) : (
        <DialogContent>Loading ...</DialogContent>
      )}
    </Dialog>
  );
}

export default RecEditDialog;

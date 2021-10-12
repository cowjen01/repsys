import { useState, useEffect } from 'react';

export function fetchItems(path) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isActive = true;

    fetch(`/api${path}`)
      .then((response) => response.json())
      .then((data) => {
        setTimeout(() => {
          if (isActive) {
            setItems(data);
          }
          setIsLoading(false);
        }, 300);
      })
      .catch((err) => {
        setIsLoading(false);
        setError(err);
      });

    return () => {
      isActive = false;
    };
  }, []);

  return { items, isLoading, error };
}

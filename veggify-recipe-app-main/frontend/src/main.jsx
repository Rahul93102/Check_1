
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import SearchPage from './components/SearchPage.jsx';

import CategoryPage from './pages/category/CategoryPage.jsx';
import Home from './pages/home/Home.jsx';
import SingleProduct from './pages/products/SingleProduct.jsx';
import Recipe from './pages/products/Recipe.jsx';
import ErrorPage from './components/ErrorPage.jsx';
import ResourcePage from './pages/resources/ResourcePage.jsx';
import ContactPage from './pages/contact/ContactPage.jsx';
import AboutPage from './pages/about/AboutPage.jsx';


const router = createBrowserRouter([
  {
    path: "/",
    element: <App/>,
    errorElement: <ErrorPage />,
    children: [
      {
        path: "/",
        element: <Home/>,
      },
      {
        path: 'categories/:category',
        element: <CategoryPage />,
      },
      {
        path: 'search',
        element: <SearchPage />,
      },
      {
        path: "/items/:id",
        element: <SingleProduct/>,
       loader: ({params}) => fetch(`http://localhost:5000/api/items/${params.id}`)
      },
      {
        path: "/recipes",
        element: <Recipe/>
      },
      {
        path: "/resources",
        element: <ResourcePage/>
      },
      {
        path: "/contact",
        element: <ContactPage/>
      },
      {
        path: "/about",
        element: <AboutPage/>
      }
    ],
  },
]);

ReactDOM.createRoot(document.getElementById('root')).render(
  <RouterProvider router={router} />,
)

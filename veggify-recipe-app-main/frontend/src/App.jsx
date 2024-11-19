import { Link, Outlet } from 'react-router-dom';
import './App.css'
import Header from './components/herader/Header';
import Footer from './components/Footer';

function App() {


  return (
    <div className='max-w-screen-2xl mx-auto'>
    <Header />
    <div className="min-h-[calc(100vh-136px)]">
        <Outlet />
      </div>
    <Footer/>
  </div>
  )
}

export default App
